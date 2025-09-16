from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List
import numpy as np
import pandas as pd

def normalize_data_val(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, cat_columns: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #split numerical and categorical features
    train_features_numerical, train_features_one_hot_encoding = _split_one_hot_encoding_and_numerical_data(X_train, cat_columns)
    val_features_numerical, val_features_one_hot_encoding = _split_one_hot_encoding_and_numerical_data(X_val, cat_columns)
    test_features_numerical, test_features_one_hot_encoding = _split_one_hot_encoding_and_numerical_data(X_test, cat_columns)
    
    #scale numerical and one-hot encoded categorical features as well as getting column names
    train_features_numerical, val_features_numerical, test_features_numerical, num_columns = _scale_numerical_data(train_features_numerical, val_features_numerical, test_features_numerical)
    train_features_one_hot_encoding, val_features_one_hot_encoding, test_features_one_hot_encoding, one_hot_encoding_columns = _one_hot_encode_categorical_data(train_features_one_hot_encoding, val_features_one_hot_encoding, test_features_one_hot_encoding, cat_columns)
    
    #concatenate the columns, the scaled numerical futures and the one-hot encoded categorical features
    columns = num_columns + one_hot_encoding_columns
    X_train_scaled = pd.DataFrame(np.hstack([train_features_numerical, train_features_one_hot_encoding]), columns=columns)
    X_val_scaled = pd.DataFrame(np.hstack([val_features_numerical, val_features_one_hot_encoding]), columns=columns)
    X_test_scaled = pd.DataFrame(np.hstack([test_features_numerical, test_features_one_hot_encoding]), columns=columns)
    return X_train_scaled, X_val_scaled, X_test_scaled

def _split_one_hot_encoding_and_numerical_data(features, one_hot_encoding_columns):
    features_numerical = features.drop(columns=one_hot_encoding_columns)
    features_one_hot_encoding = features[one_hot_encoding_columns]
    return features_numerical, features_one_hot_encoding
        
def _scale_numerical_data(num_features_train, num_features_val, num_features_test):
    scaler = StandardScaler()
    scaler.fit(num_features_train)
    features_scaled_train = scaler.transform(num_features_train)
    features_scaled_val = scaler.transform(num_features_val)
    features_scaled_test = scaler.transform(num_features_test)
    return features_scaled_train, features_scaled_val, features_scaled_test, num_features_train.columns.tolist()

def _one_hot_encode_categorical_data(cat_features_train, cat_features_val, cat_features_test, cat_columns):
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(cat_features_train)
    cat_features_train = ohe.transform(cat_features_train)
    cat_features_val = ohe.transform(cat_features_val)
    cat_features_test = ohe.transform(cat_features_test)
    one_hot_encoding_columns = ohe.get_feature_names_out(cat_columns).tolist()
    return cat_features_train, cat_features_val, cat_features_test, one_hot_encoding_columns
