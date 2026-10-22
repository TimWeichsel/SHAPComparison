import pandas as pd
from ucimlrepo import fetch_ucirepo
from data.data_splitter import split_data
from data.data_normalizer import normalize_data

def fetch_abalone(normalize: bool = True, test_split: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Fetch the Abalone dataset from UCI repository, split it into training/test set and optionally normalize the datasets

    Parameters:
        normalize -- Whether to normalize the data (default: True = normalize data)
        test_split -- Fraction of data that is in the test set (default: 0.2 = 20% test set)
    
    Returns:
        Pandas DataFrame: Training features
        Pandas DataFrame: test features
        Pandas Series: training targets
        Pandas Series: test targets
    """
    abalone = fetch_ucirepo(id=1) #id=1 is the Abalone dataset
    abalone_X = abalone.data.features
    abalone_y = abalone.data.targets

    abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test = split_data(abalone_X, abalone_y, test_split=test_split)
    
    if normalize:
        abalone_X_train, abalone_X_test = normalize_data(abalone_X_train, abalone_X_test, ["Sex"])

    return abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test