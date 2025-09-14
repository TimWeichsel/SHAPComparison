import pandas as pd
from ucimlrepo import fetch_ucirepo
from data.data_splitter import split_data
from data.data_normalizer import normalize_data

def fetch_wine_quality(normalize: bool = True, test_split: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Fetch the Wine Quality dataset from UCI repository, split it into training/test set, update target values starting from 0 and optionally normalize the datasets

    Parameters:
        normalize -- Whether to normalize the data (default: True = normalize data)
        test_split -- Fraction of data that is in the test set (default: 0.2 = 20% test set)
    
    Returns:
        Pandas DataFrame: Training features
        Pandas DataFrame: Test features
        Pandas Series: Training targets
        Pandas Series: Test targets
    """
    wine_quality = fetch_ucirepo(id=186)
    wine_quality_X = wine_quality.data.features
    wine_quality_y = wine_quality.data.targets
    wine_quality_y = wine_quality_y-3 #category values are initially [3 4 5 6 7 8 9]

    wine_quality_X_train, wine_quality_X_test, wine_quality_y_train, wine_quality_y_test = split_data(wine_quality_X, wine_quality_y, test_split=test_split)
    if normalize:
            wine_quality_X_train, wine_quality_X_test = normalize_data(wine_quality_X_train, wine_quality_X_test, [])
    
    return wine_quality_X_train, wine_quality_X_test, wine_quality_y_train, wine_quality_y_test