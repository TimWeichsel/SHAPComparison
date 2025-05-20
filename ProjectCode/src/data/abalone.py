import pandas as pd
from ucimlrepo import fetch_ucirepo
from data.data_splitter import split_data
from data.data_normalizer import normalize_data

def fetch_abalone(normalize: bool = True, test_split: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    abalone = fetch_ucirepo(id=1)
    abalone_X = abalone.data.features
    abalone_y = abalone.data.targets
    
    #Split the data into train and test sets
    abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test = split_data(abalone_X, abalone_y, test_split=test_split)
    
    if normalize:
        abalone_X_train, abalone_X_test = normalize_data(abalone_X_train, abalone_X_test, ["Sex"])
        
    return abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test