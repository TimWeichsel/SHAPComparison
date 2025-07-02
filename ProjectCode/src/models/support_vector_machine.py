from sklearn import svm
from config import Seed

def train_SVMregressor (X_train, y_train) -> svm.SVR:
    regr = svm.SVR()
    regr.fit(X_train, y_train)
    return regr
    
def train_SVMclassifier (X_train, y_train) -> svm.SVC:
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    return clf

