import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
data_A_dict = get_data("A", 0)

data_D_X_train = data_D_dict["X_train"]
data_W_X_train = data_W_dict["X_train"]
data_A_X_train = data_A_dict["X_train"]


# Source for the following code: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
df = data_B_X_train
num_cols = df.select_dtypes(include=np.number).columns
cols_to_exclude = [col for col in num_cols if set(df[col].dropna().unique()) <= {0, 1}]
filtered_df = df.drop(columns=cols_to_exclude)
corr = filtered_df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
'''


def average_absolute_correlation(df):
    num_cols = df.select_dtypes(include=np.number).columns
    cols_to_exclude = [col for col in num_cols if set(df[col].dropna().unique()) <= {0, 1}]
    filtered_df = df.drop(columns=cols_to_exclude)
    corr_matrix = filtered_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return upper_tri.stack().mean()

print(average_absolute_correlation(data_D_X_train))

print(average_absolute_correlation(data_W_X_train))

print(average_absolute_correlation(data_A_X_train))

print(average_absolute_correlation(data_B_X_train))

def max_absolute_correlation(df):
    num_cols = df.select_dtypes(include=np.number).columns
    cols_to_exclude = [col for col in num_cols if set(df[col].dropna().unique()) <= {0, 1}]
    filtered_df = df.drop(columns=cols_to_exclude)
    corr_matrix = filtered_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return upper_tri.stack().max()'''
