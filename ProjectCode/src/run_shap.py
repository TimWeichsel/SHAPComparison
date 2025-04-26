import pickle
import pandas as pd
import time
import datetime
from data.data_loader import get_data

#SHAP imports
from models.model_loader import get_model
from shap_approximations.kernel_shap import create_kernel_shap_explainer
from shap_approximations.sample_shap import create_sample_shap_explainer
from shap_approximations.copula_shap import get_copula_shap_values

def run_and_save_shap_values(data_name: str, model_name: str, approximation_name: str, nsamples: int, number_of_runs: int = 0):
    df_dict, model = __prepare_shap_calulation(data_name, model_name)
    if number_of_runs > 0:
        for run_id in range(number_of_runs):
            shap_information = __run_shap_values(data_name, model_name, approximation_name, nsamples, df_dict, model, run_id+1)
            __save_shap_values(shap_information, approximation_name, data_name, model_name, nsamples, run_id+1)
    else:
        shap_information = __run_shap_values(data_name, model_name, approximation_name, nsamples, df_dict, model)
        __save_shap_values(shap_information, approximation_name, data_name, model_name, nsamples)
    
def __prepare_shap_calulation(data_name: str, model_name: str):
    reg_classification_indicator = __check_regression_classification(model_name)
    df_dict = get_data(data_name, reg_classification_indicator)
    model = get_model(data_name, model_name)
    return df_dict, model

def __run_shap_values(data, model_name, approximation, nsamples, df_dict, model, run_id = 0):
    start_time = time.time()
    shap_values = __calculate_SHAP_approximation(model, df_dict, approximation, nsamples)
    end_time =  time.time()
    total_time = end_time - start_time
    return __create_shap_information(total_time, shap_values, model_name, data, approximation, nsamples, df_dict["X_test"], df_dict["y_test"], df_dict["X_train"].columns, datetime.datetime.now(), run_id)

def __save_shap_values(shap_information, approximation_name, data_name, model_name, nsamples, run_id = 0):
    if run_id:
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}_{run_id}.pkl"

    else:
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}.pkl"
    file = open(file_name, "wb")
    pickle.dump(shap_information, file)
    file.close()
            
def __check_regression_classification(model) -> int:
    match model:
        case "LR" | "RR" | "SR" | "XR":
            return 0
        case "LO" | "RC" | "SC" | "XC":
            return 1
        case _:
            raise ValueError("Invalid model. Choose 'LR', 'LO', 'RR', 'RC', 'SR', 'SC', 'XR' or 'XC'")
    

def __calculate_SHAP_approximation(model, df_dict, approximation, nsamples):
    X_train = pd.DataFrame(df_dict["X_train"])
    X_test = pd.DataFrame(df_dict["X_test"], columns=X_train.columns)
    shap_values = __get_SHAP_values(approximation, model, X_train, X_test, nsamples)
    return shap_values

def __get_SHAP_values(approximation: str, model, X_train, X_test, nsamples):
    match approximation:
        case "K":
            explainer = create_kernel_shap_explainer(model=model, X_train=X_train)
            return explainer.shap_values(X_test, nsamples=nsamples)
        case "S":
            pass
            explainer = create_sample_shap_explainer(model=model, X_train=X_train)
            return explainer.shap_values(X_test, nsamples=nsamples)
        case "B":
            pass
            #return create_bounded_sampling_shap_explainer
        case "C":
            return get_copula_shap_values()
        case "G":
            pass
            #return create_gauss_copula_shap_explainer
        case "E":
            pass
            #return create_exact_shap_explainer
        case _:
            pass
            #raise ValueError("Invalid SHAP approximation. Choose 'K', 'S', 'B', 'C', 'G' or 'E'")
            
def __create_shap_information(total_time, shap_values, model_name, data, approximation, nsamples, X_test, y_test, feature_names, timestamp, run_id = 1) -> dict:
    return {
        "total_time": total_time,
        "shap_values": shap_values,
        "run_id": run_id,
        "model_name": model_name,
        "data": data,
        "approximation": approximation,
        "nsamples": nsamples,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "timestamp": timestamp,
    }