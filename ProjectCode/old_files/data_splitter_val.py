import pandas as pd
from sklearn.model_selection import train_test_split
from config import Seed


def split_data_val (X: pd.DataFrame, y: pd.DataFrame, test_split: float = 0.2, val_split: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Split the data into train-validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_split, random_state=Seed.SEED
    )
    
    #Split the train-validation data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split, random_state=Seed.SEED
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test