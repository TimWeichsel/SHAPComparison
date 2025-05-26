import shap
import numpy as np
import math
from shap.maskers import Independent

def create_sample_shap_explainer_old (model, X_train, run_id_seed) -> shap.SamplingExplainer:

    
    masker = Independent(X_train)
    
    def predict_function(data):
        print("\n🟦 [DEBUG] Aufruf von predict_function")
        print(f"🔢 Eingabe-Shape: {data.shape}")
        
        if data.shape[0] == 0:
            print("⚠️ WARNUNG: Leeres Array an model.predict() übergeben!")
            return np.zeros((0,))

        print("📋 Maskierte Beispiele (X_masked):")
        for i, row in enumerate(data):
            print(f"  ▶️ {i}: {row}")
        
        result = model.predict(data)
        print(f"✅ Rückgabe-Shape: {result.shape}")
        return result
    
    sample_shap = shap.SamplingExplainer(model=predict_function, data=X_train, masker=masker, seed=run_id_seed)
    #predict_function = lambda data: model.predict(data)
    #sample_shap = shap.SamplingExplainer(model=predict_function, data=X_train, seed=run_id_seed)
    return sample_shap


def create_sample_shap_explainer (model_function, X_train, X_test, n_samples, run_id_seed) -> shap.SamplingExplainer:
    rng = np.random.default_rng(run_id_seed)
    shap_values = []
    n_features = X_train.shape[1]
    for x in X_test:
        current_shap_value = np.zeros(n_features)
        for index_feauture in range(n_features):
            current_shap_contribution = 0
            for sample_index in n_samples:
                x_i = x.copy()
                x_no_i = x.copy()
                permutation = rng.permutation(n_features)
                index_in_permutation = list(permutation).index(index_feauture) #position of the feature in the permutation
                random_data_point_index = rng.integers(0, X_train.shape[0]) 
                random_data_point = X_train[random_data_point_index]
                x_i[index_in_permutation+1:] = random_data_point[index_in_permutation+1:] #replace all features after choosen feature with the valuees of the random data point
                x_no_i[index_in_permutation:] = random_data_point[index_in_permutation:] #replace all features and the choosen feature with the valuees of the random data point
                current_shap_contribution += model_function(x_i) - model_function(x_no_i)
    return sample_shap