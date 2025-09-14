import rbo
import numpy as np
from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import save_result

def compute_rbo_score(data_name: str, model_name: str, approximation_name: str, nsamples: int, p_weight: float=0.9) -> float:    
    '''
    Compute the Degree of Importance Score with Rank-Biased Overlap (RBO) between Exact SHAP and specific SHAP approximations (data, model, approximation type and number of samples)
    The result is an average over all data points
    The results are saved directly with to the evaluations/results.pkl file
    
    Parameters:
        data_name -- name of the dataset for the SHAP values used to compute RBO score
        model_name -- name of the model for the SHAP values used to compute RBO score
        approximation_name -- name of the approximation for the SHAP values used to compute RBO score
        nsamples -- number of samples used in the SHAP approximation to compute RBO score
        p_weight -- weight parameter for the RBO calculation (default: 0.9)
    
    Returns:
        Float: Mean Degree of Importance Score (Rank-Biased Overlap)
    '''
    shap_values = get_shap_values(data_name, model_name, approximation_name, nsamples)
    exact_values = get_shap_values(data_name, model_name, "E")
    rbo_scores = []
    for data_point_index in range(shap_values.shape[0]):
        shap_values_ranked = np.argsort(-abs(shap_values[data_point_index]))
        exact_values_ranked = np.argsort(-abs(exact_values[data_point_index]))
        rbo_scores.append(rbo.RankingSimilarity(exact_values_ranked, shap_values_ranked).rbo(p=p_weight))
        
    rbo_mean_score = np.mean(rbo_scores)
    save_result(data_name, model_name, approximation_name, nsamples, f"RBO_Score_{p_weight}", rbo_mean_score)
    return rbo_mean_score



