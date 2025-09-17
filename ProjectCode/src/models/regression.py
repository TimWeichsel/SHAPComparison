from sklearn.linear_model import LinearRegression
import pandas as pd

def train_LinRegression (X_train: pd.DateFrame, y_train:pd.Series) -> LinearRegression:
    '''
    Create and train a Linear Regression model using scikit-learn
    
    Parameters:
        X_train -- Features of Train Set
        y_train -- Target of Train Set
    
    Returns:
        Trained Linear Regression model
    '''
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg



