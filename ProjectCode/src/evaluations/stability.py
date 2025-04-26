from math import sqrt
import numpy as np
from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import save_result

def compute_stability_score(data_name, model_name, approximation_name, nsamples, number_of_runs):
    shap_values = []
    for run_id in range(number_of_runs):
        shap_values.append(get_shap_values(data_name, model_name, approximation_name, nsamples, run_id+1))
    shap_array = np.array(shap_values) #shap values array (round_index, data_point_index, feature_index)
    mean_data_point = shap_array.mean(axis=0) #mean shap value array (data_point_index, feature_index) 
    sd_data_point = shap_array.std(axis=0) #standard deviasion shap value array (data_point_index, feature_index)
    Z = 1.96 #z value for 95% confidence interval
    lower_bound = mean_data_point - Z * sd_data_point/sqrt(number_of_runs)
    upper_bound = mean_data_point + Z * sd_data_point/sqrt(number_of_runs)
    ci_size_mean_per_point = upper_bound-lower_bound #confidence interval per point (data_point_index, feature_index)
    ci_size_mean_per_feature = ci_size_mean_per_point.mean(axis=0) #mean confidence interval per feature (feature_index)
    stability_score = ci_size_mean_per_feature.mean(axis=0) #mean interval = stability score
    save_result(data_name, model_name, approximation_name, nsamples, number_of_runs, "Stability_Score", stability_score)
    return stability_score
