import shap
import numpy as np
import pandas as pd
from shap.maskers import Independent

def create_sample_shap_explainer (model: object, X_train: pd.DataFrame, run_id_seed: int) -> shap.SamplingExplainer:
    '''
    Creates Sample SHAP explainer from SHAP library for a given model and training features (using Independent masker = standard Sample SHAP)
    
    Parameters:
        model -- ML model object
        X_train -- Features of Train Set
        run_id_seed -- seed for SHAP computation
    
    Returns:
        Sample SHAP explainer
    '''

    masker = Independent(X_train)
    
    def predict_function(data):
        if (data.shape[0] == 0):
            raise SystemExit("Data is empty.") # 
        result = model.predict(data)
        return result
    
    sample_shap_explainer = shap.SamplingExplainer(model=predict_function, data=X_train, masker=masker, seed=run_id_seed)
    

    return sample_shap_explainer
