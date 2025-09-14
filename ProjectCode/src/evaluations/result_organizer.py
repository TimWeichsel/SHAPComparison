import pickle
import pandas as pd
import numpy as np

def save_result(data_name: str, model_name: str, approximation_name: str, nsamples: int, category: str, result: float, number_of_runs: int|None = None) -> None:
    '''
    Save evaluation results in evaluations/results.pkl file
    
    Parameters:
        data_name -- name of the dataset for the SHAP values
        model_name -- name of the model for the SHAP values
        approximation_name -- name of the approximation for the SHAP values
        nsamples -- number of samples used in the SHAP approximation
        category -- evaluation category
        result -- evaluation result to be saved
        number_of_runs -- number of SHAP approximation runs (needed only for Stability score -> default: None)
    '''
    file_name = f"evaluations/results.pkl"
    #create key for storing results
    if number_of_runs is not None:
        key = f"{approximation_name}_{data_name}_{model_name}_{nsamples}_{number_of_runs}" #key for Stability evaluation
    else:
        key = f"{approximation_name}_{data_name}_{model_name}_{nsamples}" #key for Degree of Importance and Fidelity evaluation
    try:
        file = open(file_name, "rb")
        evaluation_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        evaluation_dict = dict()

    if key not in evaluation_dict:
        evaluation_dict[key] = dict() #create new dictionary if key not in file
    evaluation_dict[key][category] = result #update result

    
    file = open(file_name, "wb") 
    pickle.dump(evaluation_dict, file)
    file.close()
    
            
def print_results(data_name: str = "", eval_method: str = None, approximation_name: str = "", model_name: str = "") -> None:
    '''
    Print evaluation results of evaluations/results.pkl file
    
    Parameters:
        data_name -- name of the dataset for the SHAP values to filter results (default: all datasets)
        eval_method -- eval method indicator ("F" = Mean Squared Error, F_MAE = Mean Absolute Error, R = RBO, S = Stability_Score, for the SHAP values to get results (default: Running_Time)
        approximation_name -- name of the approximation for the SHAP values to filter results (default: all approximations)
        model_name -- name of the model for the SHAP values to filter results (default: all models)
    '''
    if eval_method is None:
        eval_method = "Running_Time"
    else: 
       match eval_method:
            case "F": # Mean Squared Error
                eval_method = "Fidelity_Score"
            case "F_MAE":
                eval_method = "Fidelity_Score_MAE"
            case "R":
                eval_method = "RBO_Score_0.9"
            case "S":
                eval_method = "Stability_Score"
    file_name = f"evaluations/results.pkl"
    
    try:
        file = open(file_name, "rb")
        evaluation_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_name} not found")
    
    # extract all relevant results
    eval_result_dict = [(key, value[eval_method]) for key, value in evaluation_dict.items() if eval_method in value]
    
    # sort result by {approximation_name}_{data_name}_{model_name} (always 6 characters) and then by value
    eval_result_dict.sort(key=lambda x: (x[0][:6], x[1]))
    
    # print all relevant results sorted
    for aprox_key,approx_value in eval_result_dict:
        if f"{data_name}_" in aprox_key and f"{approximation_name}_" in aprox_key and f"{model_name}_" in aprox_key and not np.isnan(approx_value):
            print(f"{aprox_key}, {approx_value}")

        
def get_results() -> pd.DataFrame:
    '''
    Load all evaluation results of evaluations/results.pkl file
    
    Returns:
        Pandas DataFrame with all evaluation results
    '''
    file_name = f"evaluations/results.pkl"
    try:
        file = open(file_name, "rb")
        evaluation_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_name} not found")
    
    # convert dict to DataFrame (Keys as index, evaluation categories as columns)
    return pd.DataFrame.from_dict(evaluation_dict).T
