import pickle
import numpy as np

def get_shap_values(data_name, model_name, approximation_name, nsamples="None", run_id = 0) -> np.ndarray:
    if run_id:
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}_{run_id}.pkl"
    else:
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}.pkl"
    try:
        file = open(file_name, "rb")
        shap_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_name} not found")
    
    if approximation_name == "E":
        return shap_dict["shap_values"].values
    elif approximation_name == "C" or approximation_name == "G":
        shap_values_with_meta = shap_dict["shap_values"].sort_values("explain_id", ascending=True)
        shap_values = shap_values_with_meta.drop(columns=["explain_id", "none"])
        return shap_values.values
    else:
        return shap_dict["shap_values"]
    
def get_run_time(data_name, model_name, approximation_name, nsamples="None", run_id = 0) -> float:
    if run_id:
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}_{run_id}.pkl"
    else:
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}.pkl"
    try:
        file = open(file_name, "rb")
        shap_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_name} not found")
    

    return shap_dict["total_time"]

