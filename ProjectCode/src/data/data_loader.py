import pickle
import pandas as pd

def get_data(data: str, regression_indicator: int) -> pd.DataFrame:
    file = open("data/all_datasets.pkl", "rb")
    data_dict = pickle.load(file)
    key = ""
    match data:
        case "A":
            if regression_indicator == 1:
                raise ValueError("Invalid data set. Abalone is not a classification data set.")
            key = "Abalone_Regression"
        
        case "D":
            if regression_indicator == 1:
                raise ValueError("Invalid data set. Diabetes is not a classification data set.")
            key = "Diabetes_Regression"
        case "W":
            if regression_indicator == 0:
                key = "WineQuality_Regression"
            else:
                key = "WineQuality_Classification"
        case "B":
            if regression_indicator == 1:
                raise ValueError("Invalid data set. Diabetes is not a classification data set.")
            key = "BlogComments_Regression"
        case _:
            raise ValueError("Invalid data set.")
    if key not in data_dict:
        raise ValueError(f"Key {key} not found")
    else:
        return data_dict[key] #returns a dictionary with "X_train", "X_test", "y_train", "y_test"