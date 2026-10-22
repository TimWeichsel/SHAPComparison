import shap
import pandas as pd

def create_kernel_shap_explainer (model: object, X_train: pd.DataFrame, run_id_seed: int) -> shap.KernelExplainer:
    '''
    Creates Kernel SHAP explainer from SHAP library for a given model and training features
    
    Parameters:
        model -- ML model object
        X_train -- Features of Train Set
        run_id_seed -- seed for SHAP computation
    
    Returns:
        Kernel SHAP explainer
    '''
    predict_function = lambda data: model.predict(data)
    kernel_shap_explainer = shap.KernelExplainer(model=predict_function, data=X_train, seed=run_id_seed)
    return kernel_shap_explainer