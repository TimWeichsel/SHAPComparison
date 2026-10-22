from sklearn.ensemble import RandomForestRegressor
from config import Seed
import pandas as pd

def train_RFregressor (X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    '''
    Create and train a Random Forest Regressor model using scikit-learn
    
    Parameters:
        X_train -- Features of Train Set
        y_train -- Target of Train Set
    
    Returns:
        Trained Random Forest model
    '''
    regr = RandomForestRegressor(random_state=Seed.SEED)
    regr.fit(X_train, y_train)
    return regr