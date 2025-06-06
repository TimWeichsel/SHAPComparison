import shap
import numpy as np
import math
from shap.maskers import Independent

def create_sample_shap_explainer (model, X_train, run_id_seed) -> shap.SamplingExplainer:

    ###DELETE
    model_call_counter = {"calls": 0}
    ###DELETE END

    masker = Independent(X_train)
    
    def predict_function(data):
        if (data.shape[0] == 0):
            raise SystemExit("Data is empty.")
        
        ###DELETE
        model_call_counter["calls"] += len(data)
        ###DELETE END

        result = model.predict(data)
        return result
    
    sample_shap = shap.SamplingExplainer(model=predict_function, data=X_train, masker=masker, seed=run_id_seed)
    #predict_function = lambda data: model.predict(data)
    #sample_shap = shap.SamplingExplainer(model=predict_function, data=X_train, seed=run_id_seed)
    

    return sample_shap, model_call_counter


def create_sample_shap_explainer_test (model_function, X_train, X_test, n_samples, run_id_seed):
    rng = np.random.default_rng(run_id_seed)
    shap_values = np.zeros(len(X_test),n_features)
    n_features = X_train.shape[1]
    for x in X_test: #calculate shap values for each sample in X_test
        current_shap_value = np.zeros(n_features)
        for index_feauture in range(n_features): #Shap value calculation for each feature of the sample point
            current_shap_contribution = 0
            for sample_index in range(n_samples): #n_samples for each feature in the sample point
                x_i = x.copy()
                x_no_i = x.copy()
                permutation = rng.permutation(n_features)
                index_in_permutation = list(permutation).index(index_feauture) #position of the observed feature in the permutation
                random_data_point_index = rng.integers(0, X_train.shape[0]) 
                random_data_point = X_train[random_data_point_index]
                for permutation_feature_index in range(index_in_permutation,n_features):
                    replacing_feature_index = permutation[permutation_feature_index] 
                    if permutation_feature_index == index_in_permutation: #observed feature in the permutation equals current feature in permutation
                        x_no_i[replacing_feature_index] = random_data_point[replacing_feature_index] #replace the choosen feature with the valuees of the random data point
                    else:
                        x_i[replacing_feature_index] = random_data_point[replacing_feature_index] #replace all features after choosen feature with the valuees of the random data point
                        x_no_i[replacing_feature_index] = random_data_point[replacing_feature_index] #replace all features and the choosen feature with the valuees of the random data point
                current_shap_contribution += (model_function(x_i) - model_function(x_no_i))
            current_shap_value[index_feauture] = current_shap_contribution / n_samples
        shap_values.append(current_shap_value)
    return shap_values