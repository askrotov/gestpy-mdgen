#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen_proj12_v04.py — двухстадийная совместная аппроксимация S1 -> D1 моделью sf3
с вариантами учёта геометрии образца и КОРРЕКТНЫМ учётом числа образцов.

ЧИСЛО ОБРАЗЦОВ ЗАВИСИТ ОТ ГЕОМЕТРИИ:
    10x10 — 10 шт., 25x25 — 10 шт., 50x50 — 4 шт., 100x100 — 2 шт.
Поэтому SEM = SD/sqrt(n_j) различается по размерам, а доверительный интервал
среднего определяется распределением Стьюдента с df = n_j - 1.
Полуширина 95% ДИ в единицах SD: 0.715 (n=10), 1.591 (n=4), 8.985 (n=2).

S1 (сорбция, 0<=dt<=t0_j):
  A : M0*F3D                                   k=3
  B : (M0 + M0L*phi)*F3D                       k=4   phi = S_бок/S_полн
  C : (M0 + M0L*g)*F3D                         k=4   g = (1/L+1/W)/max_j(1/L+1/W)
  D : M0*F3D + M0L*phi*(1-exp(-t/tauL))        k=5
  E : M0_j*F3D (свободные ёмкости по размерам) k=6

D1 (сушка, tau = dt - t0_j > 0):
  pred = C0_j - [та же конструкция с M0_d, DS_d, DH_d, M0L_d, tauL_d]
  *-tied : геометрическая добавка наследуется из S1
  *-free : добавка оценивается по сушке независимо
  TIE_DS_D1=True дополнительно фиксирует DS_d = DS (суффикс _DSfix)

Подгонка ВСЕГДА во времени t (dt / tau); sqrt — только для отображения.
Все размеры подбираются ОДНОВРЕМЕННО для каждой пары (mat, time).

Выход: par / pred / stats_sizes / compare_variants / compare_summary (+ .md),
       plots_task12/.

Запуск:
  python Qwen_proj12_v04.py            # все семейства A,B,C,D,E
  python Qwen_proj12_v04.py A,B        # только A / B-tied / B-free
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

# ============================ НАСТРОЙКИ ============================
DATA_FILE = "task12.csv"
PAR_FILE  = "none_par_task12.csv"
PRED_FILE = "none_pred_task12.csv"
SIZE_FILE = "none_stats_sizes_task12.csv"
CMP_FILE  = "none_compare_variants_task12.csv"
SUM_FILE  = "none_compare_summary_task12.csv"
MD_FILE   = "none_compare_variants_task12.md"
PLOT_DIR  = "none_plots_task12"

MATERIALS  = ["БП", "СП", "УП", "СП2УП2"]
TIME_LABEL = {0: "ИС", 2: "КЛ2", 3: "КЛ3"}
SIZES_STD  = (10, 25, 50, 100)

# --- число образцов на точку по геометрии (ll, ww) ---
N_SPEC = {(10, 10): 10, (25, 25): 10, (50, 50): 4, (100, 100): 2}
N_SPEC_DEFAULT = 1

FAMILIES  = ("A", "B", "C", "D", "E")
D1_TIES   = ("tied", "free")
TIE_DS_D1 = False

# --- нормировка дескриптора C: max_j(1/L+1/W) по стандартному набору ---
G_REF_C = 1.0 / 10 + 1.0 / 10          # = 0.2 мм^-1 (опорный размер 10x10)

NNN     = 100
NNN_MAX = 2000
SERIES_MODE = "adaptive"               # "adaptive" | "legacy"

USE_PER_SIZE_T0 = True

# --- веса: "none" | "sd" | "sem" | "sem_pool" ---
WEIGHT_MODE = "none"                   # базовый режим (совместим с v03/v04)
SIGMA_FLOOR = 0.01                     # минимальная погрешность СРЕДНЕГО, %
POOL_MIN_DM = 0.2                      # порог dm для оценки относительного разброса
POOL_MIN_PTS = 5                       # минимум точек для пул-оценки по размеру

MAPE_EPS  = 0.05
T_FICK_S1 = 63.0
T_FICK_D1 = 10.0

MAXFEV   = 60000
N_STARTS = 3
MIN_PTS  = 6

FONT_NAME  = "Arial"
X_USE_SQRT = True
SHOW_T0_LINES = True
N_DENSE    = 400
ERROR_BAR  = "sd"                      # "sd" | "sem" | "ci95"
PLOT_VARIANTS = ("A", "@bic", "@aic")
WRITE_MD = True
plt.rcParams["font.size"] = 11


# ============================ РЯД sf3 ============================
_MP  = ((2.0 * np.arange(NNN_MAX) + 1.0) ** 2) * (np.pi ** 2)
_INV = 1.0 / _MP


def _series(u, D):
    """S(u,D) = sum_p exp(-(2p+1)^2 pi^2 D u)/((2p+1)^2 pi^2), u = t/x^2."""
    u = np.asarray(u, float)
    D = float(D)
    if SERIES_MODE == "legacy":
        arg = -D * np.outer(u, _MP[:NNN])
        np.clip(arg, -700.0, 700.0, out=arg)
        return np.exp(arg) @ _INV[:NNN]
    S = np.full(u.shape, 0.125, float)
    if D <= 0.0:
        return S
    nz = u > 0.0
    if not nz.any():
        return S
    uu = u[nz]
    fo = D * float(uu.min())
    n = NNN
    if fo > 0.0:
        n = int(min(NNN_MAX, max(NNN, np.ceil(1.0 / np.sqrt(fo)))))
    arg = -D * np.outer(uu, _MP[:n])
    np.clip(arg, -700.0, 0.0, out=arg)
    S[nz] = np.exp(arg) @ _INV[:n]
    return S


def fick_frac(pack, DS, DH):
    SL = _series(pack["uL"], DS)
    SW = SL if pack["same"] else _series(pack["uW"], DS)
    SH = _series(pack["uH"], DH)
    return 1.0 - 512.0 * SL * SW * SH


# ======================= ГЕОМЕТРИЯ / ВАРИАНТЫ =======================
def geom_descriptor(mode, L, W, H):
    L = np.asarray(L, float); W = np.asarray(W, float); H = np.asarray(H, float)
    if mode in ("B", "D"):
        return H * (L + W) / (L * W + H * (L + W))
    if mode == "C":
        return (1.0 / L + 1.0 / W) / G_REF_C
    return np.zeros(np.broadcast(L, W, H).shape, float)


def size_index(ll, ww, hh, sizes):
    m = {(int(l), int(w), int(h)): i for i, (l, w, h) in enumerate(sizes)}
    return np.array([m[(int(l), int(w), int(h))]
                     for l, w, h in zip(np.ravel(ll), np.ravel(ww), np.ravel(hh))], int)


def param_names(family, sizes):
    if family == "A":
        return ["M0", "DS", "DH"]
    if family in ("B", "C"):
        return ["M0", "DS", "DH", "M0L"]
    if family == "D":
        return ["M0", "DS", "DH", "M0L", "tauL"]
    if family == "E":
        return [f"M0_{int(l)}" for (l, w, h) in sizes] + ["DS", "DH"]
    raise ValueError(family)


def d_idx(family, n_sizes):
    return (n_sizes, n_sizes + 1) if family == "E" else (1, 2)


def edge_idx(family, n_sizes):
    if family in ("B", "C"):
        return [3]
    if family == "D":
        return [3, 4]
    if family == "E":
        return list(range(n_sizes))
    return []


def bounds_full(family, sizes):
    n = len(sizes)
    if family == "A":
        lo, hi = [0, 0, 0], [100, 50, 50]
    elif family in ("B", "C"):
        lo, hi = [0, 0, 0, 0], [100, 50, 50, 10]
    elif family == "D":
        lo, hi = [0, 0, 0, 0, 0.2], [100, 50, 50, 10, 5000]
    else:
        lo, hi = [0] * n + [0, 0], [100] * n + [50, 50]
    return np.asarray(lo, float), np.asarray(hi, float)


def make_pack(t, ll, ww, hh, idx, mode):
    t = np.asarray(t, float)
    L = np.asarray(ll, float); W = np.asarray(ww, float); H = np.asarray(hh, float)
    uL = t / (L * L); uW = t / (W * W); uH = t / (H * H)
    return {"t": t, "uL": uL, "uW": uW, "uH": uH,
            "same": bool(np.array_equal(uL, uW)),
            "g": geom_descriptor(mode, L, W, H),
            "idx": np.asarray(idx, int)}


def model_family(family, p, pack, n_sizes):
    iDS, iDH = d_idx(family, n_sizes)
    F = fick_frac(pack, p[iDS], p[iDH])
    if family == "A":
        return p[0] * F
    if family in ("B", "C"):
        return (p[0] + p[3] * pack["g"]) * F
    if family == "D":
        tau = max(float(p[4]), 1e-9)
        return p[0] * F + p[3] * pack["g"] * (-np.expm1(-pack["t"] / tau))
    return p[pack["idx"]] * F


def amp_by_size(family, p, sizes):
    L = np.array([s[0] for s in sizes], float)
    W = np.array([s[1] for s in sizes], float)
    H = np.array([s[2] for s in sizes], float)
    n = len(sizes)
    if family == "A":
        return np.full(n, float(p[0]))
    if family in ("B", "C", "D"):
        return float(p[0]) + float(p[3]) * geom_descriptor(family, L, W, H)
    return np.asarray(p[:n], float)


def stage_model(family, stage, p, t, ll, ww, hh, sizes, C0=0.0):
    """Публичная точка входа (используется в qw_make_report_figs_v04.py)."""
    t = np.atleast_1d(np.asarray(t, float))
    L, W, H = (np.broadcast_to(np.asarray(v, float), t.shape) for v in (ll, ww, hh))
    idx = size_index(L, W, H, sizes)
    pack = make_pack(t, L, W, H, idx, family)
    m = model_family(family, np.asarray(p, float), pack, len(sizes))
    if stage == "S1":
        return m
    return np.broadcast_to(np.asarray(C0, float), t.shape) - m


# ==================== ЧИСЛО ОБРАЗЦОВ / ПОГРЕШНОСТИ ====================
def n_spec(ll, ww):
    return np.array([N_SPEC.get((int(l), int(w)), N_SPEC_DEFAULT)
                     for l, w in zip(np.ravel(ll), np.ravel(ww))], float)


def pooled_rel_sd(dm, sd, ll, ww):
    """Пул-оценка относительного разброса c_j по каждому размеру.
    Медиана по времени устойчива даже при n=2 (каждая SD имеет 1 df)."""
    dm = np.asarray(dm, float); sd = np.asarray(sd, float)
    ll = np.asarray(ll); ww = np.asarray(ww)
    rel = sd / np.maximum(np.abs(dm), 1e-6)
    ok = np.abs(dm) > POOL_MIN_DM
    c_glob = float(np.median(rel[ok])) if ok.sum() >= POOL_MIN_PTS else float(np.median(rel))
    out = np.empty_like(dm)
    for (l, w) in {(int(a), int(b)) for a, b in zip(ll, ww)}:
        m = (ll == l) & (ww == w)
        mm = m & ok
        c = float(np.median(rel[mm])) if mm.sum() >= POOL_MIN_PTS else c_glob
        out[m] = c * np.maximum(np.abs(dm[m]), 1e-6)
    return out


def error_columns(g):
    """Возвращает dict с n, sd, sem, sd_pool, sem_pool, ci95_mean."""
    dm = g["dm_avg"].values.astype(float)
    sd = g["dm_std"].values.astype(float)
    n = n_spec(g["ll"].values, g["ww"].values)
    sem = sd / np.sqrt(n)
    sd_pool = pooled_rel_sd(dm, sd, g["ll"].values, g["ww"].values)
    sem_pool = sd_pool / np.sqrt(n)
    tq = np.array([t_dist.ppf(0.975, max(int(k) - 1, 1)) for k in n])
    return {"n": n, "sd": sd, "sem": sem,
            "sd_pool": sd_pool, "sem_pool": sem_pool,
            "ci95_mean": tq * sem}


def sigma_for(err):
    """sigma для взвешивания / chi2 согласно WEIGHT_MODE."""
    key = {"none": "sd", "sd": "sd", "sem": "sem", "sem_pool": "sem_pool"}[WEIGHT_MODE]
    return np.maximum(err[key], SIGMA_FLOOR)


# ============================ ПОДГОНКА ============================
def starts_s1(family, sizes, y, ll, ww, hh):
    m = max(float(np.max(y)), 0.5)
    out = []
    for i in range(N_STARTS):
        ds = [0.05, 0.20, 0.50][i % 3]
        dh = [0.02, 0.05, 0.01][i % 3]
        if family == "A":
            out.append([m, ds, dh])
        elif family in ("B", "C"):
            out.append([m, ds, dh, [0.3, 0.6, 0.1][i % 3]])
        elif family == "D":
            out.append([m, ds, dh, [0.3, 0.6, 0.1][i % 3], [40.0, 150.0, 15.0][i % 3]])
        else:
            mj = []
            for (l, w, h) in sizes:
                msk = (ll == l) & (ww == w) & (hh == h)
                mj.append(max(float(np.max(y[msk])) if msk.any() else m, 0.5))
            out.append(mj + [ds, dh])
    return out


def starts_d1(family, sizes, target, pS):
    m = max(float(np.max(target)), 0.1)
    n = len(sizes)
    iDS, iDH = d_idx(family, n)
    out = []
    for i in range(N_STARTS):
        p = np.array(pS, float).copy()
        if family == "E":
            p[:n] = m
        else:
            p[0] = m
            if family in ("B", "C", "D"):
                p[3] = [0.3, 0.6, 0.1][i % 3]
            if family == "D":
                p[4] = [40.0, 150.0, 15.0][i % 3]
        p[iDS] = [0.10, 0.50, 2.00][i % 3]
        p[iDH] = [0.05, 0.10, 0.20][i % 3]
        out.append(p)
    return out


def _fit_family(family, pack, target, sigma, starts, free_idx, fixed, lo, hi, n_sizes):
    free_idx = list(free_idx)
    base = np.zeros(len(lo), float)
    for k, v in fixed.items():
        base[k] = float(v)
    dummy = np.zeros(np.size(target), float)
    use_w = (WEIGHT_MODE != "none")

    def unpack(pf):
        p = base.copy(); p[free_idx] = pf; return p

    def f(_x, *pf):
        return model_family(family, unpack(np.asarray(pf, float)), pack, n_sizes)

    best, last_err = None, None
    for st in starts:
        st = np.asarray(st, float)
        p0 = np.clip(st[free_idx], lo[free_idx] + 1e-12, hi[free_idx] - 1e-12)
        try:
            pf, cov = curve_fit(f, dummy, target, p0=p0,
                                bounds=(lo[free_idx], hi[free_idx]),
                                sigma=(sigma if use_w else None),
                                absolute_sigma=True, maxfev=MAXFEV)
        except Exception as e:
            last_err = e
            continue
        p = unpack(pf)
        sse = float(np.sum((model_family(family, p, pack, n_sizes) - target) ** 2))
        if best is None or sse < best[0]:
            best = (sse, p, cov)
    if best is None:
        raise RuntimeError(f"ни один старт не удался: {type(last_err).__name__}: {last_err}")
    return best[1], best[2]


def param_se(cov, p, free_idx):
    se = np.full(len(p), np.nan)
    if cov is not None and np.all(np.isfinite(cov)):
        se[list(free_idx)] = np.sqrt(np.abs(np.diag(cov)))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(np.abs(p) > 0, 100.0 * se / np.abs(p), np.nan)
    return se, rel


def cov_diag(cov):
    if cov is None or cov.ndim != 2 or not np.all(np.isfinite(cov)):
        return np.nan, np.nan
    d = np.sqrt(np.abs(np.diag(cov)))
    if np.any(d <= 0):
        return np.nan, np.nan
    C = cov / np.outer(d, d)
    k = C.shape[0]
    off = C[~np.eye(k, dtype=bool)]
    rmax = float(np.max(np.abs(off))) if off.size else np.nan
    try:
        cond = float(np.linalg.cond(C))
    except Exception:
        cond = np.nan
    return rmax, cond


# ============================ СТАТИСТИКИ ============================
def fit_stats(y, pred, sigma, k, sigma_alt=None):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    sigma = np.asarray(sigma, float)
    n = y.size
    res = y - pred
    sse = float(np.sum(res ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    dof = n - k
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    r2adj = (1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
             if np.isfinite(r2) and (n - k - 1) > 0 else np.nan)
    rmse = float(np.sqrt(sse / n)) if n else np.nan
    span = y.max() - y.min() if n else np.nan
    chi2 = float(np.sum((res / sigma) ** 2))
    m = np.abs(y) > MAPE_EPS
    mape = float(100.0 * np.mean(np.abs(res[m] / y[m]))) if m.any() else np.nan
    if n > 0 and sse > 0:
        aic = n * np.log(sse / n) + 2 * k
        aicc = aic + 2 * k * (k + 1) / (n - k - 1) if (n - k - 1) > 0 else np.nan
        bic = n * np.log(sse / n) + k * np.log(n)
    else:
        aic = aicc = bic = np.nan
    r = float(np.corrcoef(y, pred)[0, 1]) if n > 2 else np.nan
    out = {"N": n, "k": k, "dof": dof, "SSE": sse,
           "R2": r2, "R2_adj": r2adj, "RMSE": rmse,
           "MAE": float(np.mean(np.abs(res))), "MedAE": float(np.median(np.abs(res))),
           "MAPE_pct": mape,
           "NRMSE_pct": float(100.0 * rmse / span) if span and span > 0 else np.nan,
           "bias": float(np.mean(res)), "max_abs_err": float(np.max(np.abs(res))),
           "chi2": chi2, "chi2_red": chi2 / dof if dof > 0 else np.nan,
           "AIC": aic, "AICc": aicc, "BIC": bic, "Pearson_r": r}
    if sigma_alt is not None:
        c2 = float(np.sum((res / np.asarray(sigma_alt, float)) ** 2))
        out["chi2_sem"] = c2
        out["chi2_red_sem"] = c2 / dof if dof > 0 else np.nan
    return out


def size_stats(y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    res = y - pred
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum(res ** 2))
    return {"N": y.size, "R2": 1.0 - sse / sst if sst > 0 else np.nan,
            "RMSE": float(np.sqrt(sse / y.size)), "MAE": float(np.mean(np.abs(res))),
            "bias": float(np.mean(res)), "max_abs_err": float(np.max(np.abs(res)))}


def rmse_mask(y, pred, mask):
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((np.asarray(y)[mask] - np.asarray(pred)[mask]) ** 2)))


# ============================ ДАННЫЕ ============================
def load_task12(path=DATA_FILE):
    df = pd.read_csv(path)
    df = df[df["mat"].isin(MATERIALS)].copy()
    for c in ["dt", "t0", "ll", "ww", "hh", "dm_avg", "dm_std"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["dt", "dm_avg", "ll", "ww", "hh"])
    df["t0"] = df["t0"].fillna(0.0)
    df["dm_std"] = df["dm_std"].fillna(0.0).abs()
    root = np.sqrt(df["dt"].clip(lower=0.0))
    if "sqrt_dt" not in df.columns:
        df["sqrt_dt"] = root
    else:
        df["sqrt_dt"] = pd.to_numeric(df["sqrt_dt"], errors="coerce")
        bad = (~np.isfinite(df["sqrt_dt"])) | ((df["sqrt_dt"] - root).abs() > 1e-6)
        if bad.any():
            print(f"  [warn] sqrt_dt пересчитан для {int(bad.sum())} строк")
            df.loc[bad, "sqrt_dt"] = root[bad]
    return df.sort_values(["mat", "time", "ll", "ww", "dt"]).reset_index(drop=True)


def annotate_stages(g):
    g = g.copy()
    t0_map = {}
    for (l, w, h), s in g.groupby(["ll", "ww", "hh"]):
        pos = s.loc[s["t0"] > 0, "t0"]
        t0_map[(l, w, h)] = float(pos.iloc[0]) if len(pos) else float(s["dt"].max())
    if not USE_PER_SIZE_T0:
        common = max(t0_map.values())
        t0_map = {k: common for k in t0_map}
    g["t0_size"] = [t0_map[(l, w, h)] for l, w, h in zip(g["ll"], g["ww"], g["hh"])]
    is_dry = (g["t0"] > 0) & (g["dt"] > g["t0_size"])
    g["stage"] = np.where(is_dry, "D1", "S1")
    g["tau"] = np.where(is_dry, g["dt"] - g["t0_size"], g["dt"])
    return g


# ============================ ГРАФИКИ ============================
def xval(t):
    t = np.asarray(t, float)
    return np.sqrt(np.clip(t, 0.0, None)) if X_USE_SQRT else t


def grid_sqrt(a, b, n=N_DENSE):
    if b <= a:
        return np.array([a], float)
    t_uni = np.linspace(np.sqrt(a), np.sqrt(b), n) ** 2
    t_geo = a + np.geomspace((b - a) * 1e-5, b - a, n)
    return np.unique(np.concatenate([t_uni, t_geo]))


def plot_group(g, sizes, variant, family, pS, pD, out_dir, font=FONT_NAME):
    os.makedirs(out_dir, exist_ok=True)
    mat = g["mat"].iloc[0]; time = int(g["time"].iloc[0])
    err = error_columns(g)
    yerr = err[{"sd": "sd", "sem": "sem", "ci95": "ci95_mean"}[ERROR_BAR]]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(sizes)]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=1)
    handles = []
    for (l, w, h), c in zip(sizes, colors):
        m = (g["ll"].values == l) & (g["ww"].values == w) & (g["hh"].values == h)
        s = g[m]
        order = np.argsort(s["dt"].values)
        sv = s.iloc[order]
        ye = yerr[m][order]
        t0 = float(sv["t0_size"].iloc[0])
        nj = int(N_SPEC.get((l, w), N_SPEC_DEFAULT))
        ax.errorbar(sv["sqrt_dt"] if X_USE_SQRT else sv["dt"], sv["dm_avg"], yerr=ye,
                    fmt="o", linestyle="none", ms=4, color=c, ecolor=c, elinewidth=0.9,
                    capsize=2.5, capthick=0.9, mfc=c, mec="black", mew=0.3, alpha=0.9, zorder=3)
        tg = grid_sqrt(0.0, t0)
        ax.plot(xval(tg), stage_model(family, "S1", pS, tg, l, w, h, sizes),
                color=c, ls="-", lw=1.6, zorder=2)
        d1 = sv[sv["stage"] == "D1"]
        if len(d1):
            C0 = float(stage_model(family, "S1", pS, np.array([t0]), l, w, h, sizes)[0])
            taug = grid_sqrt(0.0, float(d1["dt"].max() - t0))
            ax.plot(xval(t0 + taug), stage_model(family, "D1", pD, taug, l, w, h, sizes, C0=C0),
                    color=c, ls="--", lw=1.6, zorder=2)
        handles.append(Line2D([], [], marker="o", ls="none", color=c, mec="black", ms=7,
                              label=f"{l}x{w}x{h} (n={nj})"))
    if SHOW_T0_LINES:
        for i, t0 in enumerate(sorted(g["t0_size"].unique())):
            ax.axvline(xval(t0), color="0.5", ls=":", lw=1.1, alpha=0.85, zorder=1)
            ax.text(xval(t0), 1.02 + 0.055 * (i % 2), f"t0={t0:g}",
                    transform=ax.get_xaxis_transform(), fontsize=9, color="0.4",
                    ha="left", fontname=font)
    ax.set_xlabel("√dt, √сут" if X_USE_SQRT else "Время dt, сутки", fontname=font, fontsize=13)
    ax.set_ylabel("dm, %", fontname=font, fontsize=13)
    ax.set_title(f"{mat} — {TIME_LABEL.get(time, time)} ({time} лет), вариант {variant}",
                 fontname=font, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(left=0.0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontname(font)
    st_h = [Line2D([], [], color="0.2", ls="-", lw=1.8, label="S1 — сорбция"),
            Line2D([], [], color="0.2", ls="--", lw=1.8, label="D1 — сушка")]
    leg = ax.legend(handles=st_h + handles, title="Стадия / размер (n образцов)",
                    loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9,
                    title_fontsize=10, prop={"family": font, "size": 9}, framealpha=0.95)
    leg.get_title().set_fontname(font)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sf3_{mat}_time{time}_{variant}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================ СХЕМЫ ВЫВОДА ============================
PAR_COLS = ["mat", "time", "time_label", "variant", "family", "tie", "stage",
            "t0_min", "t0_max", "N", "k", "dof",
            "M0", "DS", "DH", "M0L", "tauL", "M0_10", "M0_25", "M0_50", "M0_100",
            "d_M0", "d_DS", "d_DH", "d_M0L", "d_tauL",
            "d_M0_10", "d_M0_25", "d_M0_50", "d_M0_100",
            "rel_M0_pct", "rel_DS_pct", "rel_DH_pct", "rel_M0L_pct", "rel_tauL_pct",
            "rel_M0_10_pct", "rel_M0_25_pct", "rel_M0_50_pct", "rel_M0_100_pct",
            "amp_10", "amp_25", "amp_50", "amp_100",
            "tied_from_S1", "corr_max_abs", "cond_cov", "weight_mode",
            "R2", "R2_adj", "RMSE", "MAE", "MedAE", "MAPE_pct", "NRMSE_pct",
            "bias", "max_abs_err", "SSE", "chi2", "chi2_red",
            "chi2_sem", "chi2_red_sem", "AIC", "AICc", "BIC", "Pearson_r"]

PRED_COLS = ["mat", "time", "variant", "family", "tie", "stage", "ll", "ww", "hh",
             "n_spec", "geom_g", "amp", "dt", "sqrt_dt", "t0", "t0_size", "tau", "C0",
             "exp", "dm_std", "sem", "sd_pool", "sem_pool", "ci95_mean",
             "pred", "resid"]

SIZE_COLS = ["mat", "time", "variant", "family", "tie", "stage", "ll", "ww", "hh",
             "n_spec", "t0", "amp", "N", "R2", "RMSE", "MAE", "bias", "max_abs_err"]

CMP_COLS = ["mat", "time", "time_label", "variant", "family", "tie",
            "N_S1", "N_D1", "N_ALL", "k_S1", "k_D1", "k_ALL",
            "M0", "M0L", "M0L_d", "tauL", "tauL_d", "DS", "DH", "DS_d", "DH_d",
            "R2_S1", "RMSE_S1", "chi2_red_S1", "AIC_S1", "BIC_S1", "dAIC_S1", "dBIC_S1",
            "R2_D1", "RMSE_D1", "chi2_red_D1", "AIC_D1", "BIC_D1", "dAIC_D1", "dBIC_D1",
            "R2_ALL", "RMSE_ALL", "chi2_red_ALL", "chi2_red_sem_ALL",
            "AIC_ALL", "AICc_ALL", "BIC_ALL",
            "dAIC_ALL", "dBIC_ALL", "dAIC_vs_A", "dBIC_vs_A",
            "w_AIC", "w_BIC", "rank_AIC", "rank_BIC", "best_AIC", "best_BIC",
            "RMSE_S1_early", "RMSE_S1_late", "RMSE_D1_early", "RMSE_D1_late",
            "C0_10", "C0_25", "C0_50", "C0_100",
            "Cres_10", "Cres_25", "Cres_50", "Cres_100",
            "eta_10", "eta_25", "eta_50", "eta_100"]


def param_row_dict(names, p, se, rel):
    d = {}
    for nm, v, s, r in zip(names, p, se, rel):
        d[nm] = float(v)
        d["d_" + nm] = float(s) if np.isfinite(s) else np.nan
        d["rel_" + nm + "_pct"] = float(r) if np.isfinite(r) else np.nan
    return d


def enrich_cmp(gc):
    for tag in ("S1", "D1", "ALL"):
        gc[f"dAIC_{tag}"] = gc[f"AIC_{tag}"] - gc[f"AIC_{tag}"].min()
        gc[f"dBIC_{tag}"] = gc[f"BIC_{tag}"] - gc[f"BIC_{tag}"].min()
    for nm, col in (("w_AIC", "dAIC_ALL"), ("w_BIC", "dBIC_ALL")):
        w = np.exp(-0.5 * gc[col].values.astype(float))
        gc[nm] = w / w.sum() if w.sum() > 0 else np.nan
    if "A" in set(gc["variant"]):
        a0 = float(gc.loc[gc.variant == "A", "AIC_ALL"].iloc[0])
        b0 = float(gc.loc[gc.variant == "A", "BIC_ALL"].iloc[0])
        gc["dAIC_vs_A"] = gc["AIC_ALL"] - a0
        gc["dBIC_vs_A"] = gc["BIC_ALL"] - b0
    else:
        gc["dAIC_vs_A"] = np.nan
        gc["dBIC_vs_A"] = np.nan
    gc["rank_AIC"] = gc["AIC_ALL"].rank(method="min").astype(int)
    gc["rank_BIC"] = gc["BIC_ALL"].rank(method="min").astype(int)
    gc["best_AIC"] = gc["AIC_ALL"] == gc["AIC_ALL"].min()
    gc["best_BIC"] = gc["BIC_ALL"] == gc["BIC_ALL"].min()
    return gc


# ============================ MARKDOWN ============================
def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "да" if v else "нет"
    if isinstance(v, (float, np.floating)):
        return "—" if not np.isfinite(v) else f"{v:.4g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def md_table(df, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_md(cmp_df, sum_df, path=MD_FILE):
    L = ["# Сравнение вариантов учёта геометрии образца (sf3, S1/D1)", "",
         f"Режим взвешивания: `{WEIGHT_MODE}`; режим ряда: `{SERIES_MODE}`.", "",
         "Число образцов: " + ", ".join(f"{a}×{b} — {c}" for (a, b), c in N_SPEC.items()) + ".",
         "SEM = SD/√n различается по размерам; 95 % ДИ среднего — по Стьюденту.", "",
         "Варианты: **A** — единая ёмкость; **B** — $M_0+M_{0L}\\varphi$; "
         "**C** — $M_0+M_{0L}g$; **D** — торцевой канал; **E** — свободные ёмкости.", "",
         "`-tied` — добавка наследуется из S1, `-free` — оценивается по сушке.", "",
         "## Сводка по всем группам", "",
         md_table(sum_df, ["variant", "n_groups", "win_AIC", "win_BIC", "mean_dAIC",
                           "mean_dBIC", "mean_dAIC_vs_A", "mean_dBIC_vs_A",
                           "mean_RMSE_ALL"]), ""]
    focus = ["variant", "k_ALL", "RMSE_ALL", "R2_ALL", "chi2_red_ALL",
             "AIC_ALL", "dAIC_vs_A", "BIC_ALL", "dBIC_vs_A", "M0", "M0L", "M0L_d"]
    allc = ["variant", "k_S1", "k_D1", "k_ALL", "RMSE_S1", "RMSE_D1", "RMSE_ALL",
            "AIC_ALL", "dAIC_ALL", "BIC_ALL", "dBIC_ALL", "w_BIC",
            "RMSE_S1_early", "RMSE_S1_late"]
    for (mat, time), grp in cmp_df.groupby(["mat", "time"]):
        L += [f"## {mat} — {TIME_LABEL.get(int(time), time)} ({int(time)} лет)", "",
              "### A / B-tied / B-free", ""]
        f = grp[grp["variant"].isin(["A", "B-tied", "B-free"])]
        L += [md_table(f, focus) if len(f) else "_нет данных_", "",
              "### Все варианты (по BIC_ALL)", "",
              md_table(grp.sort_values("BIC_ALL"), allc), ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# ============================ MAIN ============================
def main():
    df = load_task12()
    print(f"Режим ряда: {SERIES_MODE}; веса: {WEIGHT_MODE}; TIE_DS_D1={TIE_DS_D1}")
    print("Число образцов:", {f"{a}x{b}": c for (a, b), c in N_SPEC.items()})
    par_rows, pred_rows, size_rows, cmp_frames = [], [], [], []

    for (mat, time), g0 in df.groupby(["mat", "time"]):
        g = annotate_stages(g0)
        sizes = sorted({(int(l), int(w), int(h)) for l, w, h in zip(g["ll"], g["ww"], g["hh"])})
        n = len(sizes)
        s1 = g[g["stage"] == "S1"].copy()
        d1 = g[g["stage"] == "D1"].copy()
        t0s = sorted(g["t0_size"].unique())
        time = int(time)
        print(f"\n=== {mat}, time={time} ({TIME_LABEL.get(time, time)}) ===")
        print("  размеры:", sizes, "| t0:", {f"{l}x{w}": float(t) for (l, w, h), t in
                                            g.groupby(["ll", "ww", "hh"])["t0_size"].first().items()})
        if len(s1) < MIN_PTS or len(d1) < MIN_PTS:
            print(f"  пропуск: мало точек (S1={len(s1)}, D1={len(d1)})")
            continue

        fams = []
        for f in FAMILIES:
            if f == "E" and any((l not in SIZES_STD) or (l != w) for l, w, h in sizes):
                print("  [warn] семейство E пропущено (нестандартные размеры)")
                continue
            fams.append(f)

        eS, eD = error_columns(s1), error_columns(d1)
        tS = s1["dt"].values.astype(float); yS = s1["dm_avg"].values.astype(float)
        sigS = sigma_for(eS); semS = np.maximum(eS["sem"], SIGMA_FLOOR)
        lS, wS, hS = s1["ll"].values, s1["ww"].values, s1["hh"].values
        iS = size_index(lS, wS, hS, sizes)
        tD = d1["dt"].values.astype(float); tauD = d1["tau"].values.astype(float)
        yD = d1["dm_avg"].values.astype(float)
        sigD = sigma_for(eD); semD = np.maximum(eD["sem"], SIGMA_FLOOR)
        lD, wD, hD = d1["ll"].values, d1["ww"].values, d1["hh"].values
        iD = size_index(lD, wD, hD, sizes)
        t0D = d1["t0_size"].values.astype(float)
        earlyS = tS <= T_FICK_S1
        earlyD = tauD <= T_FICK_D1

        meta = {"mat": mat, "time": time, "time_label": TIME_LABEL.get(time, str(time)),
                "t0_min": min(t0s), "t0_max": max(t0s)}

        def add_par(stage, variant, family, tie, names, p, se, rel, st, k, amp,
                    rmax, cond, tied_names):
            row = dict(meta)
            row.update({"variant": variant, "family": family, "tie": tie, "stage": stage,
                        "corr_max_abs": rmax, "cond_cov": cond,
                        "weight_mode": WEIGHT_MODE,
                        "tied_from_S1": ",".join(tied_names)})
            row.update(st)
            row["k"] = k
            if len(names):
                row.update(param_row_dict(names, p, se, rel))
            row.update({f"amp_{int(s[0])}": float(a) for s, a in zip(sizes, amp)})
            par_rows.append(row)

        def add_pred(stage, variant, family, tie, sub, err, pred, C0arr, amparr):
            ll, ww, hh = sub["ll"].values, sub["ww"].values, sub["hh"].values
            gg = geom_descriptor(family, ll, ww, hh)
            ii = size_index(ll, ww, hh, sizes)
            for j, (_, r) in enumerate(sub.iterrows()):
                pred_rows.append(dict(meta) | {
                    "variant": variant, "family": family, "tie": tie, "stage": stage,
                    "ll": r["ll"], "ww": r["ww"], "hh": r["hh"],
                    "n_spec": int(err["n"][j]), "geom_g": float(gg[j]),
                    "amp": float(amparr[ii[j]]),
                    "dt": r["dt"], "sqrt_dt": r["sqrt_dt"], "t0": r["t0"],
                    "t0_size": r["t0_size"], "tau": r["tau"], "C0": float(C0arr[j]),
                    "exp": r["dm_avg"], "dm_std": float(err["sd"][j]),
                    "sem": float(err["sem"][j]), "sd_pool": float(err["sd_pool"][j]),
                    "sem_pool": float(err["sem_pool"][j]),
                    "ci95_mean": float(err["ci95_mean"][j]),
                    "pred": float(pred[j]), "resid": float(r["dm_avg"] - pred[j])})

        def add_sizes(stage, variant, family, tie, sub, y, pred, amparr):
            ll, ww, hh = sub["ll"].values, sub["ww"].values, sub["hh"].values
            for si, (l, w, h) in enumerate(sizes):
                m = (ll == l) & (ww == w) & (hh == h)
                if not m.any():
                    continue
                stt = size_stats(y[m], pred[m])
                size_rows.append(dict(meta) | {
                    "variant": variant, "family": family, "tie": tie, "stage": stage,
                    "ll": l, "ww": w, "hh": h,
                    "n_spec": int(N_SPEC.get((l, w), N_SPEC_DEFAULT)),
                    "t0": float(sub["t0_size"].values[m][0]),
                    "amp": float(amparr[si]), **stt})

        # ---------- S1 ----------
        fam_store = {}
        for family in fams:
            names = param_names(family, sizes)
            lo, hi = bounds_full(family, sizes)
            packS = make_pack(tS, lS, wS, hS, iS, family)
            try:
                pS, covS = _fit_family(family, packS, yS, sigS,
                                       starts_s1(family, sizes, yS, lS, wS, hS),
                                       list(range(len(names))), {}, lo, hi, n)
            except Exception as e:
                print(f"  S1[{family}] fit error: {e}")
                continue
            predS = model_family(family, pS, packS, n)
            kS = len(names)
            stS = fit_stats(yS, predS, sigS, kS, sigma_alt=semS)
            seS, relS = param_se(covS, pS, range(kS))
            rmaxS, condS = cov_diag(covS)
            ampS = amp_by_size(family, pS, sizes)
            C0 = model_family(family, pS, make_pack(t0D, lD, wD, hD, iD, family), n)
            fam_store[family] = dict(names=names, pS=pS, predS=predS, stS=stS, kS=kS,
                                     C0=C0, lo=lo, hi=hi, ampS=ampS)
            add_par("S1", family, family, "", names, pS, seS, relS, stS, kS, ampS,
                    rmaxS, condS, [])
            add_pred("S1", family, family, "", s1, eS, predS, np.zeros(len(s1)), ampS)
            add_sizes("S1", family, family, "", s1, yS, predS, ampS)
            extra = "" if family == "A" else f", M0L={pS[3]:.4f}"
            if family == "D":
                extra += f", tauL={pS[4]:.1f}"
            print(f"  S1[{family}] k={kS} M0={pS[0]:.4f}{extra} | "
                  f"R2={stS['R2']:.4f} RMSE={stS['RMSE']:.4f} "
                  f"AIC={stS['AIC']:.2f} BIC={stS['BIC']:.2f}")

        # ---------- D1 + ALL ----------
        res = {}
        for family, fs in fam_store.items():
            for tie in (("free",) if family == "A" else D1_TIES):
                variant = family if family == "A" else f"{family}-{tie}"
                fixed = {}
                if tie == "tied":
                    for i in edge_idx(family, n):
                        fixed[i] = float(fs["pS"][i])
                if TIE_DS_D1:
                    fixed[d_idx(family, n)[0]] = float(fs["pS"][d_idx(family, n)[0]])
                    variant += "_DSfix"
                freeD = [i for i in range(len(fs["names"])) if i not in fixed]
                tied_names = [fs["names"][i] for i in sorted(fixed)]
                packD = make_pack(tauD, lD, wD, hD, iD, family)
                target = fs["C0"] - yD
                try:
                    pD, covD = _fit_family(family, packD, target, sigD,
                                           starts_d1(family, sizes, target, fs["pS"]),
                                           freeD, fixed, fs["lo"], fs["hi"], n)
                except Exception as e:
                    print(f"  D1[{variant}] fit error: {e}")
                    continue
                predD = fs["C0"] - model_family(family, pD, packD, n)
                kD = len(freeD)
                stD = fit_stats(yD, predD, sigD, kD, sigma_alt=semD)
                seD, relD = param_se(covD, pD, freeD)
                rmaxD, condD = cov_diag(covD)
                ampD = amp_by_size(family, pD, sizes)
                yA = np.concatenate([yS, yD])
                pA = np.concatenate([fs["predS"], predD])
                sA = np.concatenate([sigS, sigD])
                semA = np.concatenate([semS, semD])
                kA = fs["kS"] + kD
                stA = fit_stats(yA, pA, sA, kA, sigma_alt=semA)

                add_par("D1", variant, family, tie, fs["names"], pD, seD, relD, stD, kD,
                        ampD, rmaxD, condD, tied_names)
                add_par("ALL", variant, family, tie, [], [], [], [], stA, kA,
                        np.full(n, np.nan), np.nan, np.nan, tied_names)
                add_pred("D1", variant, family, tie, d1, eD, predD, fs["C0"], ampD)
                add_sizes("D1", variant, family, tie, d1, yD, predD, ampD)

                ps = dict(zip(fs["names"], fs["pS"]))
                pd_ = dict(zip(fs["names"], pD))
                row = dict(meta) | {"variant": variant, "family": family, "tie": tie,
                                    "N_S1": fs["stS"]["N"], "N_D1": stD["N"], "N_ALL": stA["N"],
                                    "k_S1": fs["kS"], "k_D1": kD, "k_ALL": kA,
                                    "M0": ps.get("M0", np.nan), "M0L": ps.get("M0L", np.nan),
                                    "M0L_d": pd_.get("M0L", np.nan),
                                    "tauL": ps.get("tauL", np.nan), "tauL_d": pd_.get("tauL", np.nan),
                                    "DS": ps.get("DS", np.nan), "DH": ps.get("DH", np.nan),
                                    "DS_d": pd_.get("DS", np.nan), "DH_d": pd_.get("DH", np.nan),
                                    "chi2_red_sem_ALL": stA.get("chi2_red_sem", np.nan),
                                    "RMSE_S1_early": rmse_mask(yS, fs["predS"], earlyS),
                                    "RMSE_S1_late": rmse_mask(yS, fs["predS"], ~earlyS),
                                    "RMSE_D1_early": rmse_mask(yD, predD, earlyD),
                                    "RMSE_D1_late": rmse_mask(yD, predD, ~earlyD)}
                for tag, st in (("S1", fs["stS"]), ("D1", stD), ("ALL", stA)):
                    for key in ("R2", "RMSE", "chi2_red", "AIC", "BIC"):
                        row[f"{key}_{tag}"] = st[key]
                row["AICc_ALL"] = stA["AICc"]
                for si, (l, w, h) in enumerate(sizes):
                    m = (lD == l) & (wD == w) & (hD == h)
                    if not m.any():
                        continue
                    c0 = float(fs["C0"][m][0])
                    am = float(ampD[si])
                    last = float(yD[m][np.argmax(tD[m])])
                    row[f"C0_{int(l)}"] = c0
                    row[f"Cres_{int(l)}"] = c0 - am
                    row[f"eta_{int(l)}"] = 100.0 * (c0 - last) / am if am > 0 else np.nan
                res[variant] = dict(family=family, tie=tie, pS=fs["pS"], pD=pD, row=row)

        if not res:
            continue
        gc = enrich_cmp(pd.DataFrame([r["row"] for r in res.values()]))
        cmp_frames.append(gc)

        print("  --- A / B-tied / B-free (ALL) ---")
        for _, r in gc[gc["variant"].isin(["A", "B-tied", "B-free"])].iterrows():
            print(f"    {r['variant']:9s} k={int(r['k_ALL']):2d} RMSE={r['RMSE_ALL']:.4f} "
                  f"R2={r['R2_ALL']:.4f} AIC={r['AIC_ALL']:9.2f} (Δ{r['dAIC_vs_A']:+7.2f}) "
                  f"BIC={r['BIC_ALL']:9.2f} (Δ{r['dBIC_vs_A']:+7.2f})")
        print("  --- ранжирование по BIC_ALL ---")
        for _, r in gc.sort_values("BIC_ALL").iterrows():
            print(f"    {r['variant']:12s} k={int(r['k_ALL']):2d} "
                  f"ΔAIC={r['dAIC_ALL']:7.2f} ΔBIC={r['dBIC_ALL']:7.2f} "
                  f"w_BIC={r['w_BIC']:.3f} RMSE={r['RMSE_ALL']:.4f}")

        chosen = []
        for tag in PLOT_VARIANTS:
            if tag == "@bic":
                v = gc.loc[gc["BIC_ALL"].idxmin(), "variant"]
            elif tag == "@aic":
                v = gc.loc[gc["AIC_ALL"].idxmin(), "variant"]
            else:
                v = tag
            if v in res and v not in chosen:
                chosen.append(v)
        for v in chosen:
            plot_group(g, sizes, v, res[v]["family"], res[v]["pS"], res[v]["pD"], PLOT_DIR)

    # ---------- сохранение ----------
    par_df = pd.DataFrame(par_rows)
    for c in PAR_COLS:
        if c not in par_df.columns:
            par_df[c] = np.nan
    par_df[PAR_COLS].to_csv(PAR_FILE, index=False)
    pd.DataFrame(pred_rows)[PRED_COLS].to_csv(PRED_FILE, index=False)
    pd.DataFrame(size_rows)[SIZE_COLS].to_csv(SIZE_FILE, index=False)

    if cmp_frames:
        cmp_df = pd.concat(cmp_frames, ignore_index=True)
        for c in CMP_COLS:
            if c not in cmp_df.columns:
                cmp_df[c] = np.nan
        cmp_df = cmp_df[CMP_COLS]
        cmp_df.to_csv(CMP_FILE, index=False)
        sum_df = (cmp_df.groupby("variant")
                  .agg(n_groups=("variant", "size"), win_AIC=("best_AIC", "sum"),
                       win_BIC=("best_BIC", "sum"), mean_dAIC=("dAIC_ALL", "mean"),
                       median_dAIC=("dAIC_ALL", "median"), mean_dBIC=("dBIC_ALL", "mean"),
                       median_dBIC=("dBIC_ALL", "median"),
                       mean_dAIC_vs_A=("dAIC_vs_A", "mean"),
                       mean_dBIC_vs_A=("dBIC_vs_A", "mean"),
                       mean_RMSE_ALL=("RMSE_ALL", "mean"), mean_k_ALL=("k_ALL", "mean"))
                  .reset_index().sort_values("mean_dBIC"))
        sum_df.to_csv(SUM_FILE, index=False)
        print("\n=== Сводка по всем группам (сортировка по среднему ΔBIC) ===")
        for _, r in sum_df.iterrows():
            print(f"  {r['variant']:12s} побед AIC/BIC: {int(r['win_AIC'])}/{int(r['win_BIC'])}"
                  f"  ср.ΔAIC={r['mean_dAIC']:7.2f}  ср.ΔBIC={r['mean_dBIC']:7.2f}"
                  f"  ср.ΔBIC vs A={r['mean_dBIC_vs_A']:+7.2f}")
        if WRITE_MD:
            write_md(cmp_df, sum_df)

    print("\nСохранено:")
    for f in (PAR_FILE, PRED_FILE, SIZE_FILE, CMP_FILE, SUM_FILE):
        print("  -", f)
    if WRITE_MD and cmp_frames:
        print("  -", MD_FILE)
    print(f"  - {PLOT_DIR}/  (варианты: {', '.join(PLOT_VARIANTS)})")
    print(f"\nРяд: {SERIES_MODE} (NNN={NNN}, NNN_MAX={NNN_MAX}); веса: {WEIGHT_MODE}; "
          f"усы на графиках: ±{ERROR_BAR}.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        FAMILIES = tuple(x.strip().upper() for x in sys.argv[1].split(",") if x.strip())
        print("Семейства:", FAMILIES)
    main()