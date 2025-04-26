import pandas as pd
from sklearn.model_selection import train_test_split
from config import Seed


def split_data (X: pd.DataFrame, y: pd.DataFrame, test_split: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Split the data into train-validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=Seed.SEED
    )
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    return X_train, X_test, y_train, y_test