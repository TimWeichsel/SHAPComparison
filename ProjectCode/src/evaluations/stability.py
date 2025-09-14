from math import sqrt
import numpy as np
from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import save_result
z_value = 1.96 #z value for 95% confidence interval

def compute_stability_score(data_name: str, model_name: str, approximation_name: str, nsamples: int, number_of_runs: int=10) -> float:
    '''
    Compute the Stability Score with mean confidence interval (CI) between SHAP approximations (data, model, approximation type and number of samples)
    The result shows the average size of the 95% confidence intervall over all features and averaged over all data points
    The results are saved directly with to the evaluations/results.pkl file

    Parameters:
        data_name -- name of the dataset for the SHAP values used to compute Stability score
        model_name -- name of the model for the SHAP values used to compute Stability score
        approximation_name -- name of the approximation for the SHAP values used to compute Stability score
        nsamples -- number of samples used in the SHAP approximation to compute Stability score
        number_of_runs -- number of SHAP approximation runs to be compared to compute Stability score (default: 10)
    
    Returns:
        Float: Stability Score (95% mean confidence interval of features)    
    ''' 
    shap_values = []
    for run_id in range(number_of_runs):
        shap_values.append(get_shap_values(data_name, model_name, approximation_name, nsamples, run_id+1))
    shap_array = np.array(shap_values) #convert to numpy array (round_index, data_point_index, feature_index)
    mean_data_point = shap_array.mean(axis=0) #mean SHAP value across runs(data_point_index, feature_index) 
    sd_data_point = shap_array.std(axis=0) #standard deviation of SHAP value across runs (data_point_index, feature_index)
    Z = z_value
    lower_bound = mean_data_point - Z * sd_data_point/sqrt(number_of_runs)
    upper_bound = mean_data_point + Z * sd_data_point/sqrt(number_of_runs)
    ci_size_mean_per_point = upper_bound-lower_bound #confidence interval (CI) per data point and feature (data_point_index, feature_index)
    ci_size_mean_per_feature = ci_size_mean_per_point.mean(axis=0) #mean CI per feature (feature_index)
    stability_score = ci_size_mean_per_feature.mean(axis=0) #mean CI aross all features = Stability score
    save_result(data_name, model_name, approximation_name, nsamples, "Stability_Score", stability_score, number_of_runs=number_of_runs)
    return stability_score
