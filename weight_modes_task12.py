#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сравнение режимов взвешивания none / sem / sem_pool по файлам v04."""

import numpy as np
import pandas as pd

MODES = [
    ("none", "none_"),
    ("sem", "sem_"),
    ("sem_pool", "sem_pool_")
]

ORDER = {"СП": 0, "УП": 1, "СП2УП2": 2, "БП": 3}
LABEL = {0: "ИС", 2: "КЛ2", 3: "КЛ3"}
SIZES = (10, 25, 50, 100)


def admissible(r):
    """Фильтр физической допустимости."""
    amp = np.array([r[f"amp_{s}"] for s in SIZES], float)
    mono = bool(np.all(np.diff(amp) <= 1e-9)) or bool((amp.max() - amp.min()) < 0.10)
    
    ok = lambda v, lo, hi: (not np.isfinite(v)) or (lo <= v <= hi)
    
    return (mono
            and ok(r.get("DH", np.nan), 0.005, 0.15)
            and ok(r.get("M0L", np.nan), -1e9, 9.989)
            and ok(r.get("tauL", np.nan), -1e9, 4998.9)
            and ok(abs(r.get("corr_max_abs", np.nan)), 0, 0.989)
            and ok(r.get("cond_cov", np.nan), 0, 999.9))


rows, recs, chi = [], [], []

for name, pfx in MODES:
    par = pd.read_csv(f"{pfx}par_task12.csv")
    cmp_ = pd.read_csv(f"{pfx}compare_variants_task12.csv")
    
    s1 = par[par.stage == "S1"].set_index(["mat", "time", "family"])
    
    for (mat, tm), g in cmp_.groupby(["mat", "time"]):
        fam_s1 = {(m, t, f): row for (m, t, f), row in s1.iterrows()}
        
        g2 = g.copy()
        g2["adm"] = [admissible(fam_s1[(mat, tm, r.family)]) if r.family in "ABCDE" else True 
                     for _, r in g2.iterrows()]
        
        best = g2.loc[g2.BIC_ALL.idxmin()]
        adm = g2[g2.adm]
        rec = adm.loc[adm.BIC_ALL.idxmin()] if len(adm) else best
        pa = par[(par.mat == mat) & (par.time == tm) & (par.variant == "A") & (par.stage == "S1")].iloc[0]
        a = g2[g2.variant == "A"].iloc[0]
        q2 = g2[g2.adm]
        
        rows.append(dict(mode=name, mat=mat, time=tm, state=LABEL.get(tm, str(tm)),
                         M0_A=pa.M0, RMSE_S1=pa.RMSE, chi2_red_S1=pa.chi2_red,
                         chi2_red_sem_S1=pa.chi2_red_sem,
                         RMSE_ALL=a.RMSE_ALL, dBIC_A=a.dBIC_vs_A, rank_A=int(a.rank_BIC),
                         best=best.variant, dBIC_best=best.dBIC_vs_A,
                         rec=rec.variant, RMSE_rec=rec.RMSE_ALL, dBIC_rec=rec.dBIC_vs_A,
                         n_adm=int(q2.adm.sum())))
        recs.append((name, mat, tm, rec.variant))
        chi.append(a.chi2_red_sem_ALL / a.chi2_red_ALL if a.chi2_red_ALL else np.nan)

df = pd.DataFrame(rows)
df.to_csv("weight_modes_task12.csv", index=False)

print("Сводка по режимам:")
print(df.pivot_table(index=["mat", "time"], columns="mode",
                     values=["M0_A", "RMSE_S1", "best", "rank_A", "rec", "RMSE_rec", "dBIC_rec"],
                     aggfunc="first").to_string())

print("\nСамопроверка chi2_red_sem/chi2_red:")
print(f"  среднее {np.nanmean(chi):.2f} (ожидается ≈6.55)")

print("\nСмена рекомендации по режимам:")
r = pd.DataFrame(recs, columns=["mode", "mat", "time", "rec"])
w = r.pivot_table(index=["mat", "time"], columns="mode", values="rec", aggfunc="first")
w["устойчиво"] = w.nunique(axis=1) == 1
print(w.to_string())

print("\nУстойчивые группы (одинаковый победитель во всех режимах):")
print(w[w["устойчиво"]].index.tolist())