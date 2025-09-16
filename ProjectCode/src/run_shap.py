import pickle
import pandas as pd
import time
import datetime
from data.data_loader import get_data
from config import Seed

#SHAP imports
from models.model_loader import get_model
from shap_approximations.kernel_shap import create_kernel_shap_explainer
from shap_approximations.sample_shap import create_sample_shap_explainer
from shap_approximations.exact_shap import create_exact_shap_explainer
from shap_approximations.copula_shap import get_copula_shap_values
from shap_approximations.gauss_shap import get_gauss_shap_values

from evaluations.result_organizer import save_result

def run_and_save_shap_values(data_name: str, model_name: str, approximation_name: str, nsamples: int, number_of_runs: int = 0)-> None:
    '''Computes the SHAP values for the specified data, model, approximation and number of samples
    
    Parameters:
        data_name -- data name indicator ("A"=Abalone, "D"=Diabetes, "M"=Music Origin, "W"=Wine Quality)
        model_name -- model name indicator ("LR","RR","SR","XR")
        approximation_name -- SHAP approximation indicator ("K", "S", "C", "G", "E", "C100", "G100")
        nsamples -- number of samples/coalitions for the SHAP approximation
        number_of_runs -- number of how many individual SHAP computations are performed (default: 0 = single run, else multiple runs for Stability)
    '''
    
    df_dict, model = __prepare_shap_calculation(data_name, model_name)
    if number_of_runs > 0:
        for run_id in range(number_of_runs):
            shap_information = __run_shap_values(data_name, model_name, approximation_name, nsamples, df_dict, model, run_id+1)
            __save_shap_values(shap_information, approximation_name, data_name, model_name, nsamples, run_id+1)
    else:
        shap_information = __run_shap_values(data_name, model_name, approximation_name, nsamples, df_dict, model, Seed.SEED)
        __save_shap_values(shap_information, approximation_name, data_name, model_name, nsamples)
    
def __prepare_shap_calculation(data_name: str, model_name: str) -> tuple[dict, object]:
    '''
    Load dataset and model for SHAP calculation
    
    Parameters:
        data_name -- data name indicator ("A"=Abalone, "D"=Diabetes, "M"=Music Origin, "W"=Wine Quality)
        model_name -- model name indicator ("LR","RR","SR","XR")
    
    Returns:
        Dict of Pandas DataFrames/Series: "X_train", "X_test", "y_train" and "y_test", ML model object
    '''
    
    classification_indicator = __check_classification(model_name)
    df_dict = get_data(data_name, classification_indicator)
    model = get_model(data_name, model_name)
    return df_dict, model

def __run_shap_values(data_name: str, model_name: str, approximation_name: str, nsamples: int, df_dict: dict, model: object, run_id_seed: int) -> dict:
    '''
    Compute SHAP approximation for one model and measures the runtime
    
    Parameters:
        data_name -- data name indicator ("A"=Abalone, "D"=Diabetes, "M"=Music Origin, "W"=Wine Quality)
        model_name -- model name indicator ("LR","RR","SR","XR")
        approximation_name -- SHAP approximation indicator ("K", "S", "C", "G", "E", "C100", "G100")
        nsamples -- number of samples/coalitions for the SHAP approximation
        df_dict -- Dict of Pandas DataFrames/Series: "X_train", "X_test", "y_train" and "y_test"
        model -- ML model object
        run_id_seed -- seed for SHAP computation (either it is fixed or varies across multiple runs for the Stability property)
        
    Returns:
        Dict with SHAP value information:
        shap_values -- computed SHAP values
        model_name -- model name indicator ("LR","RR","SR","XR")
        data_name -- data name indicator ("A"=Abalone, "D"=Diabetes, "M"=Music Origin, "W"=Wine Quality)
        approximation_name -- SHAP approximation indicator ("K", "S", "C", "G", "E", "C100", "G100")
        nsamples -- number of samples/coalitions for the SHAP approximation
        X_test --  Features of Test Set
        y_test -- Target Values of Test Set
        feature_names -- Column names of features
        timestamp -- Time when SHAP calculation was started
        run_id -- identifier for multiple runs or 1 if single run)
    '''
    
    start_time = time.time()
    shap_values = __calculate_SHAP_approximation(model, df_dict, approximation_name, nsamples, run_id_seed)
    end_time =  time.time()
    total_time = end_time - start_time
    return __create_shap_information(total_time, shap_values, model_name, data_name, approximation_name, nsamples, df_dict["X_test"], df_dict["y_test"], df_dict["X_train"].columns, datetime.datetime.now(), run_id_seed)

def __save_shap_values(shap_information, approximation_name, data_name, model_name, nsamples, run_id = 0) -> None:
    '''
    Saves the SHAP values in shap_approximations/shap_values/ folder and calls save_result to save the runtime
    
    Parameters:
        Dict with SHAP value information
        model_name -- model name indicator ("LR","RR","SR","XR")
        data_name -- data name indicator ("A"=Abalone, "D"=Diabetes, "M"=Music Origin, "W"=Wine Quality)
        approximation_name -- SHAP approximation indicator ("K", "S", "C", "G", "E", "C100", "G100")
        nsamples -- number of samples/coalitions for the SHAP approximation
        run_id -- identfier for multiple runs or 1 if single run)
    '''
    
    runtime = shap_information["total_time"]
    if run_id:
        save_result(data_name, model_name, approximation_name, nsamples,"Running_Time", result=runtime, number_of_runs=run_id)
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}_{run_id}.pkl"
        
    else:
        save_result(data_name, model_name, approximation_name, nsamples,"Running_Time", result=runtime)
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}.pkl"
    file = open(file_name, "wb")
    pickle.dump(shap_information, file)
    file.close()
            
def __check_classification(model_name) -> bool:
    '''
    Checks if the model is a classification model or regression model.
    
    Parameters:
        model_name -- model name indicator ("LR","RR","SR","XR")
        
    Returns:
        bool: False, if valid regression model; ValueError if other model, since classification not yet implemented
    
    
    '''
    match model_name:
        case "LR" | "RR" | "SR" | "XR":
            return False
        case _:
            raise ValueError("Classification not yet implemented.")
    

def __calculate_SHAP_approximation(model: object, df_dict: dict, approximation_name: str, nsamples: int, run_id_seed: int) -> object:
    '''
    Uses model information to compute SHAP values
    
    Parameters:
        model -- ML model object
        df_dict -- Dict of Pandas DataFrames/Series: "X_train", "X_test", "y_train" and "y_test"
        approximation_name -- SHAP approximation indicator ("K", "S", "C", "G", "E", "C100", "G100")
        nsamples -- number of samples/coalitions for the SHAP approximation
        run_id_seed -- seed for SHAP computation (either it is fixed or varies across multiple runs for the Stability property)
    
    Returns:
        SHAP values of the specified approximation on the model
    
    '''
    X_train = pd.DataFrame(df_dict["X_train"])
    y_train = pd.DataFrame(df_dict["y_train"])
    X_test = pd.DataFrame(df_dict["X_test"], columns=X_train.columns)
    shap_values = __get_SHAP_values(approximation_name, model, X_train, X_test, nsamples, y_train, run_id_seed)
    return shap_values

def __get_SHAP_values(approximation_name: str, model:object, X_train: pd.DataFrame, X_test: pd.DataFrame, nsamples: int, y_train: pd.DataFrame, run_id_seed: int) -> object:
    '''
    Computes the SHAP values for the specified approximation on the given model and data
    
    Parameters:
        approximation_name -- SHAP approximation indicator ("K", "S", "C", "G", "E", "C100", "G100")
        model -- ML model object
        X_train -- Features of Train Set
        X_test -- Features of Test set
        y_train -- Target Values of Train Set
        nsamples -- number of samples/coalitions for the SHAP approximation
        run_id_seed -- seed for SHAP computation (either it is fixed or varies across multiple runs for the Stability property)
    
    Returns:
        SHAP values of the specified approximation on the model
    
    '''
    match approximation_name:
        case "K":
            explainer = create_kernel_shap_explainer(model=model, X_train=X_train, run_id_seed=run_id_seed)
            return explainer.shap_values(X_test, nsamples=nsamples)
        case "S":
            explainer = create_sample_shap_explainer(model=model, X_train=X_train, run_id_seed=run_id_seed)
            return explainer.shap_values(X_test, nsamples=nsamples)
        case "C":
            return get_copula_shap_values(model=model, X_train=X_train, X_test=X_test, y_train=y_train, n_samples=nsamples, run_id_seed=run_id_seed)
        case "C100":
            return get_copula_shap_values(model=model, X_train=X_train, X_test=X_test, y_train=y_train, n_samples=nsamples, run_id_seed=run_id_seed, mc_samples=1000)
        case "G":
            return get_gauss_shap_values(model=model, X_train=X_train, X_test=X_test, y_train=y_train, n_samples=nsamples, run_id_seed=run_id_seed)
        case "G100":
            return get_gauss_shap_values(model=model, X_train=X_train, X_test=X_test, y_train=y_train, n_samples=nsamples, run_id_seed=run_id_seed, mc_samples=1000)
        case "E":
            explainer = create_exact_shap_explainer(model=model, X_train=X_train)
            return explainer(X_test, max_evals=2**20) # shap values -> limit to 2^20 evaluations
        case _:
            raise ValueError("Invalid SHAP approximation. Choose 'K', 'S', 'C', 'G', 'E', 'C100', 'G100'")
            
def __create_shap_information(total_time: float, shap_values: object, model_name: str, data_name: str, approximation_name: str, nsamples: int, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list[str], timestamp: datetime.datetime, run_id:int = 1) -> dict:
    '''
    Creates dictionary with all provided information
    
    Parameters:
        total_time -- Runtime of SHAP calculation
        shap_values -- computed SHAP values
        model_name -- model name indicator ("LR","RR","SR","XR")
        data_name -- data name indicator ("A"=Abalone, "D"=Diabetes, "M"=Music Origin, "W"=Wine Quality)
        approximation_name -- SHAP approximation indicator ("K", "S", "C", "G", "E", "C100", "G100")
        nsamples -- number of samples/coalitions for the SHAP approximation
        X_test --  Features of Test Set
        y_test -- Target Values of Test Set
        feature_names -- Column names of features
        timestamp -- Time when SHAP calculation was started
        run_id -- identifier for multiple runs or 1 if single run
    
    Returns:
        Dict with all SHAP information given as parameters
    '''
    
    return {
        "total_time": total_time,
        "shap_values": shap_values,
        "run_id": run_id,
        "model_name": model_name,
        "data": data_name,
        "approximation": approximation_name,
        "nsamples": nsamples,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "timestamp": timestamp,
    }