from external.shapr.python.shaprpy.explain import explain
import rpy2

MC_SAMPLES = 10 #fixed number for this project

def get_copula_shap_values (model, X_train, X_test, y_train, n_samples, run_id_seed):
    phi_0 = y_train.mean().iloc[0] #according to shapr: "phi0 = Numeric. The prediction value for unseen data, i.e. an estimate of the expected prediction without conditioning on any features. Typically we set this value equal to the mean of the response variable in our training data, but other choices such as the mean of the predictions in the training data are also reasonable.
    phi_0_R = rpy2.robjects.FloatVector([phi_0]) #convert to R float, needeed for the shapr function
    
    explanation = explain(
    model = model,
    x_train = X_train,
    x_explain = X_test,
    approach = 'copula',
    max_n_coalitions = n_samples, #number of coalitions
    n_MC_samples = MC_SAMPLES, #number of samples for each coalition
    phi0=phi_0_R,
    iterative = rpy2.robjects.NULL, #basic copula shap has no iterative calculation
    seed = run_id_seed
    )
    return explanation["shapley_values_est"]

