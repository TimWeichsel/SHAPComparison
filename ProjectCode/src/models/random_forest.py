from sklearn.ensemble import RandomForestRegressor
from config import Seed

def train_RFregressor (X_train, y_train) -> RandomForestRegressor:
    regr = RandomForestRegressor(random_state=Seed.SEED)
    regr.fit(X_train, y_train)
    return regr