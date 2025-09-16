from sklearn import svm

def train_SVMregressor (X_train, y_train) -> svm.SVR:
    regr = svm.SVR()
    regr.fit(X_train, y_train)
    return regr


