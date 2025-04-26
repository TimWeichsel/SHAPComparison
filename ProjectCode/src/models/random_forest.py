from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from config import Seed

def train_RFregressor (X_train, y_train) -> RandomForestRegressor:
    regr = RandomForestRegressor(random_state=Seed.SEED)
    regr.fit(X_train, y_train)
    return regr
    
    
def train_RFclassifier (X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(random_state=Seed.SEED)
    clf.fit(X_train, y_train)
    return clf