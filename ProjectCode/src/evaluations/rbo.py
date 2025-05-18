import rbo
import numpy as np
from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import save_result

def compute_rbo_score(data_name, model_name, approximation_name, nsamples, p_weight=0.9):    
    shap_values = get_shap_values(data_name, model_name, approximation_name, nsamples)
    exact_values = get_shap_values(data_name, model_name, "E", nsamples)
    shap_values_ranked = np.argsort(-abs(shap_values[0]))
    exact_values_ranked = np.argsort(-abs(exact_values.values[0]))
    print(f"Shap size: {shap_values.shape}")
    print(f"Shap size: {exact_values.values.shape}")
    print(f"Shap values: {abs(shap_values[0])}")
    print(f"Exact values: {abs(exact_values.values[0])}")
    print(f"Shap values ranked: {shap_values_ranked}")
    print(f"Exact values ranked: {exact_values_ranked}")
    
    #EACH DATA POINT CALCULATE RBO!
    #rbo.RankingSimilarity(list(exact_values), list(shap_values_ranked)).rbo(p_weight)
    #print(rbo)

