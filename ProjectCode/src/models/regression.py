from sklearn.linear_model import LinearRegression

def train_LinRegression (X_train, y_train) -> LinearRegression:
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg



