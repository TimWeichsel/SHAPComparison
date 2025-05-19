import rbo
import numpy as np
from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import save_result

def compute_rbo_score(data_name, model_name, approximation_name, nsamples, p_weight=0.9):    
    shap_values = get_shap_values(data_name, model_name, approximation_name, nsamples)
    exact_values = get_shap_values(data_name, model_name, "E", nsamples)
    rbo_scores = []
    for data_point_index in range(shap_values.shape[0]):
        shap_values_ranked = np.argsort(-abs(shap_values[data_point_index]))
        exact_values_ranked = np.argsort(-abs(exact_values.values[data_point_index]))
        rbo_scores.append(rbo.RankingSimilarity(exact_values_ranked, shap_values_ranked).rbo(p=p_weight))
        
    rbo_mean_score = np.mean(rbo_scores)
    save_result(data_name, model_name, approximation_name, nsamples, f"RBO_Score_{p_weight}", rbo_mean_score)
    return rbo_mean_score



