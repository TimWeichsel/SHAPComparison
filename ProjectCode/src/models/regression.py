from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from config import Seed

def train_LinRegression (X_train, y_train) -> LinearRegression:
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg
    
    
def train_LogRegression (X_train, y_train) -> LogisticRegression:
    clf = LogisticRegression(random_state=Seed.SEED)
    clf.fit(X_train, y_train)
    return clf


