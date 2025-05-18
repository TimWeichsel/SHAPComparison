import pickle
import pandas as pd

def get_shap_values(data_name, model_name, approximation_name, nsamples, run_id = 0) -> pd.DataFrame:
    if approximation_name == "exact":
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_None_None.pkl"
    elif run_id:
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}_{run_id}.pkl"
    else:
        file_name = f"shap_approximations/shap_values/{approximation_name}_{data_name}_{model_name}_{nsamples}.pkl"
    try:
        file = open(file_name, "rb")
        shap_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_name} not found")
    
    return shap_dict["shap_values"]