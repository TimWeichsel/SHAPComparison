import pandas as pd
from data.data_splitter import split_data
from data.data_normalizer import normalize_data
from sklearn.datasets import load_diabetes

def fetch_diabetes(normalize: bool = True, test_split: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    diabetes = load_diabetes(scaled=False, as_frame=True)
    diabetes_X = diabetes.data
    diabetes_y = diabetes.target
    
    diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = split_data(diabetes_X, diabetes_y, test_split=test_split)
    
    if normalize:
        diabetes_X_train, diabetes_X_test = normalize_data(diabetes_X_train, diabetes_X_test, ["sex"])
    return diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test