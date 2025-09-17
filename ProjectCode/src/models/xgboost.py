from xgboost import XGBRegressor
from config import Seed
import pandas as pd

def train_XGBregressor (X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    '''
    Create and train a XGBoost model using xgboost library
    
    Parameters:
        X_train -- Features of Train Set
        y_train -- Target of Train Set
    
    Returns:
        Trained XGBoost model
    '''
    bst = XGBRegressor(random_state=Seed.SEED)
    bst.fit(X_train, y_train)
    return bst