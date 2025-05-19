
from sklearn.metrics import mean_squared_error as mse
from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import save_result
import numpy as np

def compute_fidelity_score(data_name, model_name, approximation_name, nsamples):    
    shap_values = get_shap_values(data_name, model_name, approximation_name, nsamples)
    exact_values = get_shap_values(data_name, model_name, "E", nsamples)
    fidelity_scores = []
    for data_point_index in range(shap_values.shape[0]):
        exact_value_data_point = exact_values.values[data_point_index]
        shap_values_data_point = shap_values[data_point_index]
        fidelity_scores.append(mse(exact_value_data_point, shap_values_data_point))
        
    fidelity_mean_score = np.mean(fidelity_scores)
    save_result(data_name, model_name, approximation_name, nsamples, "Fidelity_Score", fidelity_mean_score)
    return fidelity_mean_score



