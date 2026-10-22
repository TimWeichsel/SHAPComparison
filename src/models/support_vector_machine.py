from sklearn import svm
import pandas as pd

def train_SVMregressor (X_train: pd.DataFrame, y_train: pd.Series) -> svm.SVR:
    '''
    Create and train a Support Vector Machine model using scikit-learn
    
    Parameters:
        X_train -- Features of Train Set
        y_train -- Target of Train Set
    
    Returns:
        Trained Support Vector Machine model
    '''
    regr = svm.SVR()
    regr.fit(X_train, y_train)
    return regr


