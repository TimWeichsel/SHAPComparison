
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import save_result
import numpy as np

def compute_fidelity_score_MSE(data_name: str, model_name: str, approximation_name: str, nsamples: int) -> float:
    '''
    Compute the Fidelity Score with Meand Squared Error (MSE) between Exact SHAP and specific SHAP approximations (data, model, approximation type and number of samples)
    The result is an average over all features and all data points
    The results are saved directly with to the evaluations/results.pkl file
    
    Parameters:
        data_name -- name of the dataset for the SHAP values used to compute Fidelity score
        model_name -- name of the model for the SHAP values used to compute Fidelity score
        approximation_name -- name of the approximation for the SHAP values used to compute Fidelity score
        nsamples -- number of samples used in the SHAP approximation to compute Fidelity score
    
    Returns:
        Float: Mean Fidelity Score (Average Mean Squared Error)
    '''
    shap_values = get_shap_values(data_name, model_name, approximation_name, nsamples)
    exact_values = get_shap_values(data_name, model_name, "E")
    fidelity_scores = []
    for data_point_index in range(shap_values.shape[0]):
        exact_value_data_point = exact_values[data_point_index]
        shap_values_data_point = shap_values[data_point_index]
        fidelity_scores.append(mse(exact_value_data_point, shap_values_data_point)) #mse over all features
        
    fidelity_mean_score = np.mean(fidelity_scores)
    save_result(data_name, model_name, approximation_name, nsamples, "Fidelity_Score", fidelity_mean_score)
    return fidelity_mean_score

def compute_fidelity_score_MAE(data_name: str, model_name: str, approximation_name: str, nsamples: int) -> float:
    '''
    Compute the Fidelity Score with Mean Absolut Error (MSE) between Exact SHAP and specific SHAP approximations (data, model, approximation type and number of samples)
    The result is an average over all features and all data points
    The results are saved directly with to the evaluations/results.pkl file

    Parameters:
        data_name -- name of the dataset for the SHAP values used to compute Fidelity score
        model_name -- name of the model for the SHAP values used to compute Fidelity score
        approximation_name -- name of the approximation for the SHAP values used to compute Fidelity score
        nsamples -- number of samples used in the SHAP approximation to compute Fidelity score
    
    Returns:
        Float: Mean Fidelity Score (Average Mean Squared Error)    
    ''' 
    shap_values = get_shap_values(data_name, model_name, approximation_name, nsamples)
    exact_values = get_shap_values(data_name, model_name, "E")
    fidelity_scores_mae = []
    for data_point_index in range(shap_values.shape[0]):
        exact_value_data_point = exact_values[data_point_index]
        shap_values_data_point = shap_values[data_point_index]
        fidelity_scores_mae.append(mae(exact_value_data_point, shap_values_data_point))
        
    fidelity_mean_score_mae = np.mean(fidelity_scores_mae)
    save_result(data_name, model_name, approximation_name, nsamples, "Fidelity_Score_MAE", fidelity_mean_score_mae)
    return fidelity_mean_score_mae



