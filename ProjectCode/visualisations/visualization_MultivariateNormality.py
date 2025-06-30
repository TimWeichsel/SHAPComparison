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
data_B_dict = get_data("B", 0)
data_A_dict = get_data("A", 0)

data_D_X_train = data_D_dict["X_train"]
data_W_X_train = data_W_dict["X_train"]
data_B_X_train = data_B_dict["X_train"]
data_A_X_train = data_A_dict["X_train"]

D_mulivariant = multivariate_normality(data_D_X_train, alpha=.05)
W_mulivariant = multivariate_normality(data_W_X_train, alpha=.05)
B_mulivariant = multivariate_normality(data_B_X_train, alpha=.05)
A_mulivariant = multivariate_normality(data_A_X_train, alpha=.05)
print(D_mulivariant)
print(W_mulivariant)
print(B_mulivariant)
print(A_mulivariant)
'''
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairwise_distribution(df, title):
    sns.set(style="whitegrid")
    sns.pairplot(df.sample(n=min(200, len(df))), diag_kind="kde")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Für D:
plot_pairwise_distribution(data_D_X_train, "Pairplot – D: X_train")

# Für W:
plot_pairwise_distribution(data_W_X_train, "Pairplot – W: X_train")'''
