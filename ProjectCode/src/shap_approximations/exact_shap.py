import shap

def create_exact_shap_explainer (model, X_train) -> shap.ExactExplainer:
    predict_function = lambda data: model.predict(data)
    exact_shap = shap.explainers.Exact(predict_function, X_train)
    return exact_shap