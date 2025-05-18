import rbo
import numpy as np
from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import save_result

def compute_rbo_score(data_name, model_name, approximation_name, nsamples, p_weight=0.9):    
    shap_values = get_shap_values(data_name, model_name, approximation_name, nsamples)
    exact_values = get_shap_values(data_name, model_name, "E", nsamples)
    shap_values_ranked = np.argsort(abs(shap_values[0]))
    exact_values_ranked = np.argsort(abs(exact_values[0]))
    print(f"Shap values: {shap_values_ranked}")
    print(f"Exact values: {exact_values_ranked}")
    rbo = []#rbo.RankingSimilarity(exact_values, shap_values, p_weight).rbo()
    return rbo

