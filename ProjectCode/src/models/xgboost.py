from xgboost import XGBRegressor
from config import Seed

def train_XGBregressor (X_train, y_train) -> XGBRegressor:
    bst = XGBRegressor(random_state=Seed.SEED)
    bst.fit(X_train, y_train)
    return bst