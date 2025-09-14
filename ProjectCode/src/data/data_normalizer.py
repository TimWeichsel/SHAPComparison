from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List
import numpy as np
import pandas as pd

def normalize_data(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_columns: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Normalize the training and test features with StandardScaler (numerical features) and OneHotEnoder (categorical features)
    
    Parameters:
        X_train -- Training features
        X_test -- Test features
        cat_columns -- Categorical column names
    Returns:
        Tuple of DataFrames: [Training features, Test features]
    '''
    #split numerical and categorical features
    train_features_numerical, train_features_one_hot_encoding = __split_one_hot_encoding_and_numerical_data(X_train, cat_columns)
    test_features_numerical, test_features_one_hot_encoding = __split_one_hot_encoding_and_numerical_data(X_test, cat_columns)
    
    #scale numerical and one-hot encoded categorical features as well as getting column names
    train_features_numerical, test_features_numerical, num_columns = __scale_numerical_data(train_features_numerical, test_features_numerical)
    train_features_one_hot_encoding, test_features_one_hot_encoding, one_hot_encoding_columns = __one_hot_encode_categorical_data(train_features_one_hot_encoding, test_features_one_hot_encoding, cat_columns)
    
    #concatenate the columns, the scaled numerical futures and the one-hot encoded categorical features
    columns = num_columns + one_hot_encoding_columns #column names
    X_train_scaled = pd.DataFrame(np.hstack([train_features_numerical, train_features_one_hot_encoding]), columns=columns)
    X_test_scaled = pd.DataFrame(np.hstack([test_features_numerical, test_features_one_hot_encoding]), columns=columns)
    return X_train_scaled, X_test_scaled

def __split_one_hot_encoding_and_numerical_data(features: pd.DataFrame, one_hot_encoding_columns: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Split the features into numerical and categorical (one hot encoded) Dataframes
    
    Parameters:
        features -- Training features
        one_hot_encoding_columns -- one_hot_encoding_columns
        cat_columns -- Categorical column names
    Returns:
        Tuple of DataFrames: [Training features, Test features]
    '''
    features_numerical = features.drop(columns=one_hot_encoding_columns)
    features_one_hot_encoding = features[one_hot_encoding_columns]
    return features_numerical, features_one_hot_encoding
        
def __scale_numerical_data(num_features_train, num_features_test):
    '''
    Scale the numerical features and return them with numerical column names
    
    Parameters:
        num_features_train -- Numerical training features
        num_features_test -- Numerical test features
    Returns:
        Scaled numerical training features, Scaled numerical test features, List of Numerical column names
    '''
    scaler = StandardScaler()
    scaler.fit(num_features_train)
    features_scaled_train = scaler.transform(num_features_train)
    features_scaled_test = scaler.transform(num_features_test)
    return features_scaled_train, features_scaled_test, num_features_train.columns.tolist()

def __one_hot_encode_categorical_data(cat_features_train, cat_features_test, cat_columns):
    '''
    One hot encode the categorical features and return them with column names
    
    Parameters:
        cat_features_train -- Categorical training features
        cat_features_test -- Categorical test features
    Returns:
        OHE categorical training features, OHE categorical test features, List of Categorical column names
    '''
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(cat_features_train)
    cat_features_train = ohe.transform(cat_features_train)
    cat_features_test = ohe.transform(cat_features_test)
    one_hot_encoding_columns = ohe.get_feature_names_out([str(col) for col in cat_columns]).tolist()
    return cat_features_train, cat_features_test, one_hot_encoding_columns
