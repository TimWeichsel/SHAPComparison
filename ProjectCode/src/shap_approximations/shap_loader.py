import pickle
import numpy as np

def get_shap_values(data_name: str, model_name: str, approximation_name: str, nsamples: str | int="None", run_id:int=0, instance_index:int = None) -> np.ndarray:
    '''
    Loads all SHAP values in the provided format of the library from shap_approximations/shap_values/ and converts it to a numpy array
    
    Parameters:
        data_name -- data name indicator ("A"=Abalone, "D"=Diabetes, "M"=Music Origin, "W"=Wine Quality)
        model_name -- model name indicator ("LR","RR","SR","XR")
        approximation_name -- SHAP approximation indicator ("K", "S", "C", "G", "E", "C100", "G100")
        nsamples -- number of samples/coalitions for the SHAP approximation, default: "None" for Exact SHAP
        run_id -- identifier for multiple runs or 1 if single run
        instance_index -- index when only one specific SHAP value should be returned (default: return all instances)    
    
    Returns:
        SHAP values as numpy array
    '''
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
        if instance_index is None:
            return shap_dict["shap_values"].values
        else:
            return shap_dict["shap_values"].values[instance_index]
    elif approximation_name == "C" or approximation_name == "G" or approximation_name == "C100" or approximation_name == "G100":
        shap_values_with_meta = shap_dict["shap_values"].sort_values("explain_id", ascending=True) #meta data provided by shapr package
        shap_values = shap_values_with_meta.drop(columns=["explain_id", "none"])
        
        if instance_index is None:
            return shap_values.values
        else:
            return shap_values.values[instance_index]
    else:
        if instance_index is None:
            return shap_dict["shap_values"]
        else:
            return shap_dict["shap_values"][instance_index]

