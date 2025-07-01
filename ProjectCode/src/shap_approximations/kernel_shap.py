import shap

def create_kernel_shap_explainer (model, X_train, run_id_seed) -> shap.KernelExplainer:
    
    predict_function = lambda data: model.predict(data)
    kernel_shap = shap.KernelExplainer(model=predict_function, data=X_train, seed=run_id_seed)
    
    return kernel_shap