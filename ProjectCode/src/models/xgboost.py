from xgboost import XGBRegressor, XGBClassifier
from config import Seed

def train_XGBregressor (X_train, y_train) -> XGBRegressor:
    bst = XGBRegressor(random_state=Seed.SEED)
    bst.fit(X_train, y_train)
    return bst
    
    
def train_XGBclassifier (X_train, y_train) -> XGBClassifier:
    bst = XGBClassifier(random_state=Seed.SEED)
    bst.fit(X_train, y_train)
    return bst
