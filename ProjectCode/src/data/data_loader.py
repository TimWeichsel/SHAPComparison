import pickle
import pandas as pd

def get_data(data_indicator: str, classification_indicator: bool = False) -> dict:
    """
    Load the datasets from Pickle and returns the specific dataset as a dictionary with "X_train", "X_test", "y_train" and "y_test"

    Parameters:
        data_indicator -- One letter indicator of the dataset ("A" = Abalone, "D" = Diabetes, "M" = Music, "W" = Wine Quality)
        classification_indicator -- Indicates if dataset is used for regression (False) or classification (True); (default: False)
    
    Returns:
        Dictionary consisting of Pandas DataFrames/Series: "X_train", "X_test", "y_train" and "y_test"
    """
    file = open("data/all_datasets.pkl", "rb")
    data_dict = pickle.load(file)
    key = ""
    match data_indicator:
        case "A":
            if classification_indicator:
                raise ValueError("Invalid data set. Abalone is not a classification data set.")
            key = "Abalone_Regression"
        
        case "D":
            if classification_indicator:
                raise ValueError("Invalid data set. Diabetes is not a classification data set.")
            key = "Diabetes_Regression"
        case "M":
            if classification_indicator:
                raise ValueError("Invalid data set. Music is not a classification data set.")
            key = "Music_Regression"
        case "W":
            if classification_indicator:
                key = "WineQuality_Classification"
            else:
                key = "WineQuality_Regression"
        case _:
            raise ValueError("Invalid data set.")
    if key not in data_dict:
        raise ValueError(f"Key {key} not found")
    else:
        return data_dict[key] #returns a dictionary with "X_train", "X_test", "y_train", "y_test"