import numpy as np
from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import save_result

def compute_rbo_score(data_name, model_name, approximation_name, nsamples, number_of_runs):
    shap_values = []
    for run_id in range(number_of_runs):
        shap_values.append(get_shap_values(data_name, model_name, approximation_name, nsamples, run_id+1))
    shap_array = np.array(shap_values) #shap values array (round_index, data_point_index, feature_index)
    rbo = None
    save_result(data_name, model_name, approximation_name, nsamples, "RBO_Score", rbo, number_of_runs=number_of_runs)
    return rbo
