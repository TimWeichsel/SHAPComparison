import shap

def create_kernel_shap_explainer (model, X_train, run_id_seed) -> shap.KernelExplainer:
    
    #UNCOMMENT
    #predict_function = lambda data: model.predict(data)
    #UNCOMMENT END
    
    #DELETE
    model_call_counter = {"calls": 0}
    def predict_function(data):
        model_call_counter["calls"] += len(data)
        result = model.predict(data)
        return result
    #DELETE END
    
    kernel_shap = shap.KernelExplainer(model=predict_function, data=X_train, seed=run_id_seed)
    
    return kernel_shap, model_call_counter