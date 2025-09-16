'''import pandas as pd
import shap
from data.data_splitter import split_data
from data.data_normalizer import normalize_data

def fetch_mortality(normalize: bool = True, test_split: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mortality_X, mortality_y_numpy = shap.datasets.nhanesi(display=True)
    mortality_y = pd.Series(mortality_y_numpy)
    
    for value in mortality_y:
       print(value)

    print(mortality_X.columns)
    print(mortality_X.shape)
    print(mortality_y.head())
    print(mortality_y.shape)
    abalone = fetch_ucirepo(id=1)
    abalone_X = abalone.data.features
    abalone_y = abalone.data.targets
    
    #Split the data into train and test sets
    abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test = split_data(abalone_X, abalone_y, test_split=test_split)
    
    if normalize:
        abalone_X_train, abalone_X_test = normalize_data(abalone_X_train, abalone_X_test, ["Sex"])
        
    return abalone_X_train, abalone_X_test, abalone_y_train, abalone_y_test'''