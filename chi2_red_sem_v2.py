import pandas as pd

par = pd.read_csv("none_par_task12.csv")
a_all = par[(par.variant == "A") & (par.stage == "ALL")]

print("Проверка для варианта A, стадия ALL:")
for _, r in a_all.iterrows():
    print(f"{r.mat} {r.time_label}:")
    print(f"  chi2_red = {r.chi2_red}")
    print(f"  chi2_red_sem = {r.chi2_red_sem}")
    print(f"  ratio = {r.chi2_red_sem / r.chi2_red if r.chi2_red else 'N/A'}")
    print(f"  N = {r.N}, k = {r.k}")