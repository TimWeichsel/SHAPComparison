import pandas as pd
import numpy as np
from pingouin import multivariate_normality
import sys
import os
import numpy as np
from pathlib import Path

project_start_point = Path(__file__).resolve().parent.parent
src_path = project_start_point / "src"
sys.path.append(str(src_path))

from data.data_loader import get_data
os.chdir(project_start_point / "src")

data_D_dict = get_data("D", 0)
data_W_dict = get_data("W", 0)
data_M_dict = get_data("M", 0)
data_A_dict = get_data("A", 0)

data_D_X_train = data_D_dict["X_train"]
data_W_X_train = data_W_dict["X_train"]
data_M_X_train = data_M_dict["X_train"]
data_A_X_train = data_A_dict["X_train"]

D_mulivariant = multivariate_normality(data_D_X_train, alpha=.05)
W_mulivariant = multivariate_normality(data_W_X_train, alpha=.05)
M_mulivariant = multivariate_normality(data_M_X_train, alpha=.05)
A_mulivariant = multivariate_normality(data_A_X_train, alpha=.05)

print(D_mulivariant)
print(W_mulivariant)
print(M_mulivariant)
print(A_mulivariant)

import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairwise_distribution(df, title):
    sns.set(style="whitegrid")
    sns.pairplot(df.sample(n=min(200, len(df))), diag_kind="kde")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
'''
# Für D:
plot_pairwise_distribution(data_D_X_train, "Pairplot – D: X_train")

# Für W:
plot_pairwise_distribution(data_W_X_train, "Pairplot – W: X_train")

# Für M:
plot_pairwise_distribution(data_M_X_train, "Pairplot – M: X_train")

# Für A:
plot_pairwise_distribution(data_A_X_train, "Pairplot – A: X_train")
'''

import numpy as np
import pandas as pd
from scipy.stats import shapiro, normaltest, anderson

def normality_report(df: pd.DataFrame, alpha: float = 0.05, max_n: int = 5000, random_state: int = 0):
    """
    Erstellt einen Normalitäts-Report pro numerischem Feature.
    Beinhaltet:
      - Shapiro-Wilk: W, p, -log10(p)
      - D'Agostino-Pearson: K2, p  (optional interpretativ, bleibt model-agnostisch)
      - Anderson-Darling: A2, kritischer 5%-Wert (nur Statistik + Schwelle)
      - Schiefe und Exzess-Kurtosis
    Sortiert nach -log10(p_shapiro) absteigend.
    """
    df_num = df.select_dtypes(include=[np.number])
    rows = []
    rng = np.random.default_rng(random_state)

    for col in df_num.columns:
        x = df_num[col].dropna().to_numpy()
        n = x.size
        if n < 3:
            rows.append({
                "feature": col, "n": n,
                "shapiro_W": np.nan, "shapiro_p": np.nan, "minus_log10_p": np.nan,
                "dagostino_K2": np.nan, "dagostino_p": np.nan,
                "anderson_A2": np.nan, "anderson_crit_5pct": np.nan,
                "skew": np.nan, "kurt_excess": np.nan,
                "shapiro_reject@alpha": np.nan
            })
            continue

        # subsample bei sehr großen n (Shapiro ist offiziell für n<=5000 empfohlen)
        if n > max_n:
            idx = rng.choice(n, size=max_n, replace=False)
            x = x[idx]
            n = x.size

        # Shapiro
        W, p_s = shapiro(x)
        minus_log10_p = -np.log10(p_s) if (p_s > 0) else np.inf

        # D'Agostino-Pearson (optional, hilfreich zur Einordnung Art der Abweichung)
        K2, p_k2 = normaltest(x)

        # Anderson-Darling
        ad_res = anderson(x, dist='norm')
        A2 = ad_res.statistic
        # den 5%-kritischen Wert herausziehen (Index 2)
        crit_5 = ad_res.critical_values[2] if len(ad_res.critical_values) >= 3 else np.nan

        # Schiefe & Exzess-Kurtosis
        skew = pd.Series(x).skew()
        kurt_excess = pd.Series(x).kurt()  # Pandas gibt Exzess-Kurtosis (Fisher) zurück

        rows.append({
            "feature": col, "n": n,
            "shapiro_W": W, "shapiro_p": p_s, "minus_log10_p": minus_log10_p,
            "dagostino_K2": K2, "dagostino_p": p_k2,
            "anderson_A2": A2, "anderson_crit_5pct": crit_5,
            "skew": skew, "kurt_excess": kurt_excess,
            "shapiro_reject@alpha": (p_s < alpha)
        })

    rep = pd.DataFrame(rows).sort_values("minus_log10_p", ascending=False).reset_index(drop=True)

    # Gesamtflagge: alle marginal normal nach Shapiro?
    all_marginal_normal = not rep["shapiro_reject@alpha"].fillna(False).any()

    return rep, all_marginal_normal

# Beispiel: für deine vier Datensätze
D_rep, D_all_normal = normality_report(data_D_X_train, alpha=0.05)
W_rep, W_all_normal = normality_report(data_W_X_train, alpha=0.05)
M_rep, M_all_normal = normality_report(data_M_X_train, alpha=0.05)
A_rep, A_all_normal = normality_report(data_A_X_train, alpha=0.05)

print("D all marginally normal (Shapiro):", D_all_normal)
print("W all marginally normal (Shapiro):", W_all_normal)
print("M all marginally normal (Shapiro):", M_all_normal)
print("A all marginally normal (Shapiro):", A_all_normal)

# Top-10 deutlich abweichende Features je Datensatz (höchste Evidenz gegen Normalität):
print("\nD (Top 10):\n", D_rep.head(10))
print("\nW (Top 10):\n", W_rep.head(10))
print("\nM (Top 10):\n", M_rep.head(10))
print("\nA (Top 10):\n", A_rep.head(10))


#print(D_shapiro)