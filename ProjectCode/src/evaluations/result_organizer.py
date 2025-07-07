import pickle
import pandas as pd
import numpy as np

def save_result(data_name, model_name, approximation_name, nsamples, category, result, number_of_runs = None):
    file_name = f"evaluations/results.pkl"
    if number_of_runs is not None:
        key = f"{approximation_name}_{data_name}_{model_name}_{nsamples}_{number_of_runs}"
    else:
        key = f"{approximation_name}_{data_name}_{model_name}_{nsamples}"
    try:
        file = open(file_name, "rb")
        evaluation_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        evaluation_dict = dict()

    if key not in evaluation_dict:
        evaluation_dict[key] = dict() 
    evaluation_dict[key][category] = result

    
    file = open(file_name, "wb") 
    pickle.dump(evaluation_dict, file)
    file.close()
    
            
def print_results(data_name: str = "", eval_method: str = None, approximation_name: str = "", model_name: str = ""):
    if eval_method is None:
        eval_method = "Running_Time"
    else: 
       match eval_method:
            case "F":
                eval_method = "Fidelity_Score"
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
    
    eval_result_dict = [(key, value[eval_method]) for key, value in evaluation_dict.items() if eval_method in value]
    eval_result_dict.sort(key=lambda x: (x[0][:6], x[1]))
    for aprox_key,approx_value in eval_result_dict:
        if f"{data_name}_" in aprox_key and f"{approximation_name}_" in aprox_key and f"{model_name}_" in aprox_key and not np.isnan(approx_value):
            print(f"{aprox_key}, {approx_value}")

        
def get_results() -> pd.DataFrame:
    file_name = f"evaluations/results.pkl"
    try:
        file = open(file_name, "rb")
        evaluation_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_name} not found")
    
    return pd.DataFrame.from_dict(evaluation_dict).T
