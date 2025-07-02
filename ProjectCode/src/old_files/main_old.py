import sys
from pathlib import Path

src_path = str(Path(__file__).parent/"src")
sys.path.append(src_path)

#fetch data functions imports
from data.abalone import fetch_abalone
from data.wine_quality import fetch_wine_quality

#model functions imports
from models.regression import train_LinRegression
from models.regression import train_LogRegression
from models.random_forest import train_RFregressor
from models.random_forest import train_RFclassifier
from models.support_vector_machine import train_SVMregressor
from models.support_vector_machine import train_SVMclassifier
from models.xgboost import train_XGBregressor
from models.xgboost import train_XGBclassifier


#model evaluation imports
from sklearn.metrics import mean_squared_error, accuracy_score

#SHAP imports
from shap_approximations.kernel_shap import create_kernel_shap_explainer


def fetch_all_data(test_split: float = 0.2):
    reg_data = []
    classification_data = []
    reg_data.append(("Abalone", fetch_abalone(split=True, normalize=True, test_split=test_split)))
    reg_data.append(("Wine Quality Regression", fetch_wine_quality(split=True, normalize=True, test_split=test_split)))
    classification_data.append(("Wine Quality Classification", fetch_wine_quality(split=True, normalize=True, test_split=test_split)))
    return reg_data, classification_data

def train_all_models(reg_data,classification_data):
    trained_models = []
    for data_name, (X_train, X_test, y_train, y_test) in reg_data:
        #0=Regression, model_name, data_set, model, X_train, X_test, y_train, y_test
        trained_models.append((0,"LinRegression", data_name, train_LinRegression(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((0,"RFRegressor", data_name, train_RFregressor(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((0,"SVMRegressor", data_name, train_SVMregressor(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((0,"XGBRegressor", data_name, train_XGBregressor(X_train, y_train),X_train, X_test, y_train, y_test))

        
    for data_name, (X_train, X_test, y_train, y_test) in classification_data:
        #0=Classification, model_name, data_set, model, X_train, X_test, y_train, y_test
        trained_models.append((1,"LogRegression", data_name, train_LogRegression(X_train, y_train),X_train, X_test, y_train, y_test))       
        trained_models.append((1,"RFClassifier", data_name, train_RFclassifier(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((1,"SVMClassifier", data_name, train_SVMclassifier(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((1,"XGBClassifier", data_name, train_XGBclassifier(X_train, y_train),X_train, X_test, y_train, y_test))

    return trained_models

def evaluate_all_models(models):
    for indicator, model_name, data_set, model, _, X_test, _, y_test in models:
        if indicator == 0:
            # Regression
            y_hat = model.predict(X_test)
            mse = mean_squared_error(y_test, y_hat)
            print(f"Model: {model_name}, Data set: {data_set}, MSE: {mse}")
        else:
            # Classification
            y_hat = model.predict(X_test)
            mse = accuracy_score(y_test, y_hat)
            print(f"Model: {model_name}, Data set: {data_set}, Accuracy Score: {mse}")

    return models


def calculate_SHAP_approximations(models):
    shap_approximations = []
    for indicator, model_name, data_set, model, X_train, X_test, y_train, y_test in models:
        #title, shap_explainer, shap_values, Reg/Classification_Indicator, model_name, data_set, model, X_train, X_test, y_train, y_test
        kernel_explainer = create_kernel_shap_explainer(model=model, data=X_train)
        kernel_shap_values = kernel_explainer.shap_values(X_test)
        shap_approximations.append((f"Kernel SHAP, Model: {model_name}, Data set: {data_set}", kernel_explainer, kernel_shap_values,indicator, model_name, data_set, model, X_train, X_test, y_train, y_test))
        #shap_approximations.append((f"Sampling SHAP, Model: {model_name}, Data set: {data_set}", lambda data: model.predict(data), indicator, model_name, data_set, model, X_train, X_test, y_train, y_test))
    return shap_approximations

def main():
    print("Main has started...")
    # Fetching all data
    reg_data, classification_data = fetch_all_data()
    print("All data has been loaded...")
    #print(reg_data[0])
    
    # Train all models on each data set
    models = train_all_models(reg_data,classification_data)
    print("All models are trained...")
    
    # Evaluate all models
    evaluate_all_models(models)
    
    #calculate SHAP approximations for each model-data set combinations
    shap_approximations = calculate_SHAP_approximations(models)
    for title, shap_explainer, shap_values, indicator, model_name, data_set, model, X_train, X_test, y_train, y_test in shap_approximations:
        print(f"{title}, SHAP explainer: {shap_explainer}, SHAP values: {shap_values}")
    
if __name__ == "__main__":
    main()
    

