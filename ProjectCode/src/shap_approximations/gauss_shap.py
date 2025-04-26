from external.shapr.python.shaprpy.explain import explain

def get_gauss_shap_values (model, X_train, X_test):
    explanation = explain(
    model = model,
    x_train = X_train,
    x_explain = X_test,
    approach = 'gauss'
    )
    return explanation["shapley_values_est"]

