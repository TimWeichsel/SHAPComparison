import shap

def create_sample_shap_explainer (model, X_train) -> shap.SamplingExplainer:
    predict_function = lambda data: model.predict(data)
    sample_shap = shap.SamplingExplainer(model=predict_function, data=X_train)
    return sample_shap