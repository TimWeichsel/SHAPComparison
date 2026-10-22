import shap
import pandas as pd

def create_exact_shap_explainer (model: object, X_train:pd.DataFrame) -> shap.ExactExplainer:
    '''
    Creates Exact SHAP explainer from SHAP library for a given model and training features
    
    Prameters:
        model -- Trained model
        X_train -- Training features
        
    Returns:
        Exact SHAP explainer
    '''
    predict_function = lambda data: model.predict(data)
    exact_shap = shap.explainers.Exact(predict_function, X_train)
    return exact_shap