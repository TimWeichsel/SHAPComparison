import shap
import numpy as np
import math
from shap.maskers import Independent

def create_sample_shap_explainer (model, X_train, run_id_seed) -> shap.SamplingExplainer:


    masker = Independent(X_train)
    
    def predict_function(data):
        if (data.shape[0] == 0):
            raise SystemExit("Data is empty.")
        result = model.predict(data)
        return result
    
    sample_shap = shap.SamplingExplainer(model=predict_function, data=X_train, masker=masker, seed=run_id_seed)
    

    return sample_shap
