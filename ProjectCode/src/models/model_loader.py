import pickle
import pandas as pd

def get_model(data: str, model: str):
    file = open("models/all_models.pkl", "rb")
    model_dict = pickle.load(file)
    model_name = ""
    data_name = ""

    match model:
        case "LR":
            model_name = "LinRegression"
        case "LO":
            model_name = "LogRegression"
        case "RR":
            model_name = "RFRegressor"
        case "RC":
            model_name = "RFClassifier"
        case "SR":
            model_name = "SVMRegressor"
        case "SC":
            model_name = "SVMClassifier"
        case "XR":
            model_name = "XGBRegressor"
        case "XC":
            model_name = "XGBClassifier"
        case _:
            raise ValueError("Invalid  moodel")
        
        
    match data:
        case "A":
            data_name = "Abalone"
        case "W":
            data_name = "WineQuality"
        case _:
            raise ValueError("Invalid data set.")
        
    key = f"{model_name}_{data_name}"
    if key not in model_dict:
        raise ValueError(f"Model Key not found {key}")
    else: 
        return model_dict[key]