#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Рисунки для отчётов задач 1 и 2 — с учётом вариантов геометрии (A/B/C/D/E, tied/free).

S1 (сорбция):   X = sqrt(dt),        Y = dm
D1 (десорбция): X = sqrt(dt - t0),   Y = dm
Единая шкала ординат для ВСЕХ графиков; компактный размер.

Вход: par_task12.csv, pred_task12.csv, compare_variants_task12.csv.

Модельные кривые:
  1) приоритет — импорт расчётного модуля (param_names + stage_model);
  2) если он не найден/старый — встроенная реализация тех же семейств A/B/C/D/E.

Выход в figs_task12/:
  s1_<MAT>_y<T>_<variant>.png   — сорбция, все размеры, эксп. ±SD + модель
  d1_<MAT>_y<T>_<variant>.png   — сушка
  cmp_<MAT>_y<T>.png            — сравнение вариантов на 10x10 и 100x100 (S1)
  ic_<MAT>_y<T>.png             — ΔAIC / ΔBIC по вариантам
  amp_<MAT>_y<T>.png            — ёмкость M0_j как функция размера (S1)

Запуск:
  python qw_make_report_figs_v04.py                       # auto: A + лучший по BIC
  python qw_make_report_figs_v04.py A,B-tied,B-free       # явный список вариантов
  python qw_make_report_figs_v04.py all Qwen_proj12_v04   # все варианты + имя модуля
"""

import os
import sys
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------- НАСТРОЙКИ -------------------------
PAR_FILE = "par_task12.csv"
PRED_FILE = "pred_task12.csv"
CMP_FILE = "compare_variants_task12.csv"
OUT_DIR = "figs_task12"

CORE_MODULE = ""            # "" = автопоиск; или явное имя, напр. "Qwen_proj12_v04"
CORE_CANDIDATES = ["Qwen_proj12_v04", "Qwen_proj12_v03", "Qwen_proj12_v05",
                   "qw_fit_variants", "Qwen_proj12_variants"]
ALLOW_FALLBACK = True       # встроенная реализация моделей, если модуль не найден

FONT = ["Arial", "DejaVu Sans"]
FIGSIZE = (5.0, 3.3)
DPI = 300
YLIM = (-0.5, 3.5)          # ЕДИНАЯ шкала ординат, %
XLIM_S1 = (0.0, 16.5)       # sqrt(dt),  √сут
XLIM_D1 = (0.0, 7.5)        # sqrt(tau), √сут
N_DENSE = 400

TASKS = {1: ["СП", "УП", "СП2УП2"], 2: ["СП", "БП"]}
LAT = {"СП": "SP", "БП": "BP", "УП": "UP", "СП2УП2": "SP2UP2"}
TIME_LABEL = {0: "ИС", 2: "КЛ2", 3: "КЛ3"}

VARIANTS = "auto"           # "auto" | "all" | список ["A","B-tied","B-free"]
AUTO_INCLUDE_AIC = False
MAKE_STAGE_FIGS = True
MAKE_CMP_FIG = True
MAKE_IC_FIG = True
MAKE_AMP_FIG = True
CMP_SIZES = (10, 100)

# параметры встроенной модели (используются только в режиме fallback)
G_REF_C = 0.2
NNN = 100
NNN_MAX = 2000

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ================= ВСТРОЕННАЯ РЕАЛИЗАЦИЯ МОДЕЛЕЙ (fallback) =================
_MP = ((2.0 * np.arange(NNN_MAX) + 1.0) ** 2) * (np.pi ** 2)
_INV = 1.0 / _MP


def _fb_series(u, D):
    u = np.asarray(u, float)
    D = float(D)
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


def _fb_geom(mode, L, W, H):
    L = np.asarray(L, float); W = np.asarray(W, float); H = np.asarray(H, float)
    if mode in ("B", "D"):
        return H * (L + W) / (L * W + H * (L + W))
    if mode == "C":
        return (1.0 / L + 1.0 / W) / G_REF_C
    return np.zeros(np.broadcast(L, W, H).shape, float)


def _fb_param_names(family, sizes):
    if family == "A":
        return ["M0", "DS", "DH"]
    if family in ("B", "C"):
        return ["M0", "DS", "DH", "M0L"]
    if family == "D":
        return ["M0", "DS", "DH", "M0L", "tauL"]
    if family == "E":
        return [f"M0_{int(l)}" for (l, w, h) in sizes] + ["DS", "DH"]
    raise ValueError(f"неизвестное семейство: {family}")


def _fb_size_index(ll, ww, hh, sizes):
    m = {(int(l), int(w), int(h)): i for i, (l, w, h) in enumerate(sizes)}
    return np.array([m[(int(l), int(w), int(h))]
                     for l, w, h in zip(np.ravel(ll), np.ravel(ww), np.ravel(hh))], int)


def _fb_stage_model(family, stage, p, t, ll, ww, hh, sizes, C0=0.0):
    t = np.atleast_1d(np.asarray(t, float))
    L, W, H = (np.broadcast_to(np.asarray(v, float), t.shape) for v in (ll, ww, hh))
    p = np.asarray(p, float)
    n = len(sizes)
    iDS, iDH = (n, n + 1) if family == "E" else (1, 2)
    uL = t / (L * L); uW = t / (W * W); uH = t / (H * H)
    SL = _fb_series(uL, p[iDS])
    SW = SL if np.array_equal(uL, uW) else _fb_series(uW, p[iDS])
    SH = _fb_series(uH, p[iDH])
    F = 1.0 - 512.0 * SL * SW * SH
    g = _fb_geom(family, L, W, H)
    if family == "A":
        m = p[0] * F
    elif family in ("B", "C"):
        m = (p[0] + p[3] * g) * F
    elif family == "D":
        tau = max(float(p[4]), 1e-9)
        m = p[0] * F + p[3] * g * (-np.expm1(-t / tau))
    else:
        idx = _fb_size_index(L, W, H, sizes)
        m = p[idx] * F
    if stage == "S1":
        return m
    return np.broadcast_to(np.asarray(C0, float), t.shape) - m


def load_core():
    """Возвращает (namespace с param_names/stage_model, описание источника)."""
    names = ([CORE_MODULE] if CORE_MODULE else []) + CORE_CANDIDATES
    for nm in names:
        if not nm:
            continue
        try:
            mod = importlib.import_module(nm)
        except Exception:
            continue
        if hasattr(mod, "param_names") and hasattr(mod, "stage_model"):
            return mod, f"импорт модуля {nm} ({getattr(mod, '__file__', '?')})"
    if ALLOW_FALLBACK:
        ns = SimpleNamespace(param_names=_fb_param_names, stage_model=_fb_stage_model)
        return ns, "ВСТРОЕННАЯ реализация моделей (расчётный модуль с новым API не найден)"
    raise SystemExit("Не найден расчётный модуль с param_names/stage_model. "
                     "Укажите CORE_MODULE или разрешите ALLOW_FALLBACK.")


core, CORE_SRC = load_core()


# ------------------------- ВСПОМОГАТЕЛЬНОЕ -------------------------
def grid_sqrt(a, b, n=N_DENSE):
    if b <= a:
        return np.array([a], float)
    t_uni = np.linspace(np.sqrt(a), np.sqrt(b), n) ** 2
    t_geo = a + np.geomspace((b - a) * 1e-5, b - a, n)
    return np.unique(np.concatenate([t_uni, t_geo]))


def style(ax, xlim, xlabel, title, fs=10):
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(xlabel, fontname=FONT, fontsize=fs)
    ax.set_ylabel("dm, %", fontname=FONT, fontsize=fs)
    ax.set_title(title, fontname=FONT, fontsize=fs + 1)
    ax.grid(True, ls="--", alpha=0.35)
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=1)
    ax.tick_params(labelsize=fs - 1)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontname(FONT)


def group_sizes(sub):
    return sorted({(int(l), int(w), int(h))
                   for l, w, h in zip(sub["ll"], sub["ww"], sub["hh"])})


def params_from_row(row, family, sizes):
    names = core.param_names(family, sizes)
    missing = [nm for nm in names if nm not in row.index or not np.isfinite(float(row[nm]))]
    if missing:
        raise KeyError(f"в строке par ({family}, stage={row.get('stage')}) "
                       f"нет параметров: {missing}")
    return np.array([float(row[nm]) for nm in names], float)


def family_of_variant(par_g):
    d = par_g[par_g["stage"] == "D1"]
    return dict(zip(d["variant"], d["family"]))


def pick_variants(par_g, cmp_g):
    have = list(dict.fromkeys(par_g.loc[par_g["stage"] != "S1", "variant"]))
    if isinstance(VARIANTS, (list, tuple)):
        return [v for v in VARIANTS if v in have]
    if VARIANTS == "all":
        return have
    out = ["A"] if "A" in have else []
    if cmp_g is not None and len(cmp_g) and "BIC_ALL" in cmp_g.columns:
        v = cmp_g.loc[cmp_g["BIC_ALL"].idxmin(), "variant"]
        if v in have and v not in out:
            out.append(v)
        if AUTO_INCLUDE_AIC and "AIC_ALL" in cmp_g.columns:
            v = cmp_g.loc[cmp_g["AIC_ALL"].idxmin(), "variant"]
            if v in have and v not in out:
                out.append(v)
    return out or have[:1]


def s1_row(par_g, family):
    r = par_g[(par_g["stage"] == "S1") & (par_g["family"] == family)]
    return r.iloc[0] if len(r) else None


def d1_row(par_g, variant):
    r = par_g[(par_g["stage"] == "D1") & (par_g["variant"] == variant)]
    return r.iloc[0] if len(r) else None


# ------------------------- РИСУНКИ СТАДИЙ -------------------------
def make_stage_fig(pred_sub, prow, family, sizes, stage, mat, time, variant):
    p = params_from_row(prow, family, sizes)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(sizes)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles = []
    for (l, w, h), c in zip(sizes, colors):
        s = pred_sub[(pred_sub["ll"] == l) & (pred_sub["ww"] == w) &
                     (pred_sub["hh"] == h)].sort_values("dt")
        if s.empty:
            continue
        if stage == "S1":
            x_exp = np.sqrt(s["dt"].values.astype(float))
            tg = grid_sqrt(0.0, float(s["dt"].max()))
            y_mod = core.stage_model(family, "S1", p, tg, l, w, h, sizes)
            x_mod = np.sqrt(tg)
        else:
            tau = s["tau"].values.astype(float)
            x_exp = np.sqrt(np.clip(tau, 0.0, None))
            C0 = float(s["C0"].iloc[0])
            tg = grid_sqrt(0.0, float(tau.max()))
            y_mod = core.stage_model(family, "D1", p, tg, l, w, h, sizes, C0=C0)
            x_mod = np.sqrt(tg)
        ax.errorbar(x_exp, s["dm_avg"], yerr=s["dm_std"], fmt="o", ls="none",
                    ms=3.2, color=c, ecolor=c, elinewidth=0.7, capsize=2.0,
                    capthick=0.7, mfc=c, mec="black", mew=0.25, alpha=0.9, zorder=3)
        ax.plot(x_mod, y_mod, color=c, lw=1.4, zorder=2)
        handles.append(Line2D([], [], marker="o", ls="none", color=c, mec="black",
                              ms=5, label=f"{l}x{w}"))
    xlabel = "√dt, √сут" if stage == "S1" else "√(dt − t0), √сут"
    xlim = XLIM_S1 if stage == "S1" else XLIM_D1
    stage_ru = "сорбция" if stage == "S1" else "десорбция"
    style(ax, xlim, xlabel, f"{mat}, {TIME_LABEL.get(time, time)} — {stage_ru} [{variant}]")
    leg = ax.legend(handles=handles, title="Размер, мм",
                    loc="lower right" if stage == "S1" else "upper right",
                    fontsize=8, title_fontsize=8, framealpha=0.9,
                    prop={"family": FONT, "size": 8}, borderpad=0.4, handlelength=1.2)
    leg.get_title().set_fontname(FONT)
    fig.tight_layout(pad=0.4)
    name = f"{stage.lower()}_{LAT[mat]}_y{int(time)}_{variant}.png"
    fig.savefig(os.path.join(OUT_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return name


# ------------------------- СРАВНЕНИЕ ВАРИАНТОВ -------------------------
def make_cmp_fig(pred_g, par_g, sizes, variants, mat, time):
    pick = [(l, w, h) for (l, w, h) in sizes if l in CMP_SIZES]
    if not pick:
        return None
    fam_map = family_of_variant(par_g)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:max(len(variants), 2)]
    fig, axes = plt.subplots(1, len(pick), figsize=(FIGSIZE[0] * len(pick), FIGSIZE[1]),
                             squeeze=False)
    for ax, (l, w, h) in zip(axes[0], pick):
        s = pred_g[(pred_g["stage"] == "S1") & (pred_g["ll"] == l) &
                   (pred_g["ww"] == w) & (pred_g["hh"] == h)]
        if s.empty:
            continue
        s = s[s["family"] == s["family"].iloc[0]].sort_values("dt")
        ax.errorbar(np.sqrt(s["dt"].values), s["dm_avg"], yerr=s["dm_std"],
                    fmt="o", ls="none", ms=3.0, color="0.35", ecolor="0.65",
                    elinewidth=0.6, capsize=1.8, capthick=0.6, alpha=0.9, zorder=3)
        tg = grid_sqrt(0.0, float(s["dt"].max()))
        for v, c in zip(variants, colors):
            fam = fam_map.get(v, v)
            row = s1_row(par_g, fam)
            if row is None:
                continue
            try:
                p = params_from_row(row, fam, sizes)
            except KeyError:
                continue
            ax.plot(np.sqrt(tg), core.stage_model(fam, "S1", p, tg, l, w, h, sizes),
                    color=c, lw=1.5, label=v, zorder=2)
        style(ax, XLIM_S1, "√dt, √сут", f"{l}x{w}x{h} мм")
        leg = ax.legend(fontsize=7, title=f"{mat}, {TIME_LABEL.get(time, time)}",
                        title_fontsize=8, framealpha=0.9, loc="lower right",
                        prop={"family": FONT, "size": 7})
        leg.get_title().set_fontname(FONT)
    fig.tight_layout(pad=0.4)
    name = f"cmp_{LAT[mat]}_y{int(time)}.png"
    fig.savefig(os.path.join(OUT_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return name


def make_ic_fig(cmp_g, mat, time):
    if cmp_g is None or cmp_g.empty:
        return None
    if not {"dAIC_ALL", "dBIC_ALL", "BIC_ALL"} <= set(cmp_g.columns):
        return None
    d = cmp_g.sort_values("BIC_ALL")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - 0.2, d["dAIC_ALL"].values, width=0.4, label="ΔAIC", color="tab:blue")
    ax.bar(x + 0.2, d["dBIC_ALL"].values, width=0.4, label="ΔBIC", color="tab:orange")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.axhline(2.0, color="0.6", lw=0.7, ls=":")
    ax.axhline(10.0, color="0.6", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(d["variant"].values, rotation=45, ha="right",
                       fontname=FONT, fontsize=8)
    ax.set_ylabel("Δ критерия (отн. лучшего)", fontname=FONT, fontsize=10)
    ax.set_title(f"{mat}, {TIME_LABEL.get(time, time)} — ALL", fontname=FONT, fontsize=11)
    ax.tick_params(labelsize=8)
    for lab in ax.get_yticklabels():
        lab.set_fontname(FONT)
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.legend(fontsize=8, prop={"family": FONT, "size": 8}, framealpha=0.9)
    fig.tight_layout(pad=0.4)
    name = f"ic_{LAT[mat]}_y{int(time)}.png"
    fig.savefig(os.path.join(OUT_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return name


def make_amp_fig(par_g, sizes, mat, time):
    fams = list(dict.fromkeys(par_g.loc[par_g["stage"] == "S1", "family"]))
    if not fams:
        return None
    xs = [s[0] for s in sizes]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(fams)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for fam, c in zip(fams, colors):
        row = s1_row(par_g, fam)
        if row is None:
            continue
        ys = [float(row[f"amp_{int(l)}"]) if f"amp_{int(l)}" in row.index else np.nan
              for l in xs]
        if not np.any(np.isfinite(ys)):
            continue
        ax.plot(xs, ys, "o-", color=c, ms=4, lw=1.4, label=fam)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(l) for l in xs], fontname=FONT, fontsize=9)
    ax.set_xlabel("Размер в плане L=W, мм", fontname=FONT, fontsize=10)
    ax.set_ylabel("Ёмкость $M_{0,j}$, %", fontname=FONT, fontsize=10)
    ax.set_title(f"{mat}, {TIME_LABEL.get(time, time)} — S1", fontname=FONT, fontsize=11)
    ax.tick_params(labelsize=8)
    for lab in ax.get_yticklabels():
        lab.set_fontname(FONT)
    ax.grid(True, ls="--", alpha=0.35)
    leg = ax.legend(fontsize=8, title="Вариант", title_fontsize=8,
                    prop={"family": FONT, "size": 8}, framealpha=0.9)
    leg.get_title().set_fontname(FONT)
    fig.tight_layout(pad=0.4)
    name = f"amp_{LAT[mat]}_y{int(time)}.png"
    fig.savefig(os.path.join(OUT_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return name


# ------------------------- MAIN -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Источник модели:", CORE_SRC)
    par = pd.read_csv(PAR_FILE)
    pred = pd.read_csv(PRED_FILE)

    # v04: колонка эксперимента называется 'exp'; приводим к 'dm_avg'
    if "dm_avg" not in pred.columns:
        if "exp" in pred.columns:
            pred = pred.rename(columns={"exp": "dm_avg"})
        else:
            raise SystemExit("pred_task12.csv: нет ни 'dm_avg', ни 'exp' — "
                         "перезапустите Qwen_proj12_v04.py")

    for nm, df in (("par", par), ("pred", pred)):
        need = {"variant", "family", "stage"}
        if not need <= set(df.columns):
            raise SystemExit(f"В {nm} нет колонок {sorted(need - set(df.columns))} — "
                             "перезапустите расчётный скрипт с вариантами A/B/C/D/E")
    cmp_all = pd.read_csv(CMP_FILE) if os.path.exists(CMP_FILE) else None
    if cmp_all is None:
        print("[warn] нет", CMP_FILE, "— auto-выбор ограничится вариантом A, ic-рисунки пропущены")
    made, skipped = [], []

    for task, mats in TASKS.items():
        for mat in mats:
            if mat not in set(pred["mat"]):
                continue
            for time in sorted(pred.loc[pred["mat"] == mat, "time"].unique()):
                time = int(time)
                pg = pred[(pred["mat"] == mat) & (pred["time"] == time)]
                pa = par[(par["mat"] == mat) & (par["time"] == time)]
                cg = (cmp_all[(cmp_all["mat"] == mat) & (cmp_all["time"] == time)]
                      if cmp_all is not None else None)
                if pg.empty or pa.empty:
                    continue
                sizes = group_sizes(pg)
                variants = pick_variants(pa, cg)
                fam_map = family_of_variant(pa)
                print(f"\n{mat} — {TIME_LABEL.get(time, time)}: варианты {variants}")

                if MAKE_STAGE_FIGS:
                    for v in variants:
                        fam = fam_map.get(v, v)
                        rS = s1_row(pa, fam)
                        subS = pg[(pg["stage"] == "S1") & (pg["family"] == fam)]
                        if rS is not None and not subS.empty:
                            try:
                                made.append(make_stage_fig(subS, rS, fam, sizes, "S1",
                                                           mat, time, v))
                            except KeyError as e:
                                skipped.append(f"S1 {mat} y{time} {v}: {e}")
                                print(f"  [skip] S1 {mat} y{time} {v}: {e}")
                        rD = d1_row(pa, v)
                        subD = pg[(pg["stage"] == "D1") & (pg["variant"] == v)]
                        if rD is not None and not subD.empty:
                            try:
                                made.append(make_stage_fig(subD, rD, fam, sizes, "D1",
                                                           mat, time, v))
                            except KeyError as e:
                                skipped.append(f"D1 {mat} y{time} {v}: {e}")
                                print(f"  [skip] D1 {mat} y{time} {v}: {e}")
                if MAKE_CMP_FIG:
                    made.append(make_cmp_fig(pg, pa, sizes, variants, mat, time))
                if MAKE_IC_FIG:
                    made.append(make_ic_fig(cg, mat, time))
                if MAKE_AMP_FIG:
                    made.append(make_amp_fig(pa, sizes, mat, time))

    made = [m for m in made if m]
    print(f"\nСоздано {len(made)} рисунков в {OUT_DIR}/ (единая шкала Y = {YLIM}):")
    for nm in sorted(made):
        print("  " + nm)
    if skipped:
        print("\nПропущено:")
        for s in skipped:
            print("  " + s)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("auto", "all"):
        VARIANTS = [s.strip() for s in sys.argv[1].split(",") if s.strip()]
    elif len(sys.argv) > 1:
        VARIANTS = sys.argv[1]
    if len(sys.argv) > 2:
        CORE_MODULE = sys.argv[2]
        core, CORE_SRC = load_core()
    main()