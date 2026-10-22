from shaprpy.explain import explain
import pandas as pd
import rpy2

MC_SAMPLES = 10 #fixed number for this project

def get_gauss_shap_values (model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, n_samples: int, run_id_seed: int, mc_samples: int=MC_SAMPLES):
    '''
    Compute and return approximated SHAP values estimated via Multivariate Gaussian SHAP using the shapr package in R
    
    Parameters:
        model -- ML model object
        X_train -- Features of Train Set
        X_test -- Features of Test set
        y_train -- Target Values of Train Set
        nsamples -- number of samples/coalitions for the SHAP approximation
        run_id_seed -- seed for SHAP computation
    
    Returns:
        Multivariate Gaussian SHAP values
    '''
    phi_0 = y_train.mean().iloc[0] #according to shapr: "phi0 = Numeric. The prediction value for unseen data, i.e. an estimate of the expected prediction without conditioning on any features. Typically we set this value equal to the mean of the response variable in our training data, but other choices such as the mean of the predictions in the training data are also reasonable.
    phi_0_R = rpy2.robjects.FloatVector([phi_0]) #convert to R float, needed for the shapr function
    
    explanation = explain(
    model = model,
    x_train = X_train,
    x_explain = X_test,
    approach = 'gaussian',
    max_n_coalitions = n_samples, #number of coalitions
    n_MC_samples = mc_samples, #number of samples for each coalition
    phi0=phi_0_R,
    iterative = rpy2.robjects.NULL, #basic copula shap has no iterative calculation
    seed = run_id_seed
    )
    return explanation["shapley_values_est"]
