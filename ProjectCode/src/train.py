import pickle

#fetch data functions imports
from data.abalone import fetch_abalone
from data.wine_quality import fetch_wine_quality
from data.diabetes import fetch_diabetes
from data.music import fetch_music

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

def train_models(test_split: float = 0.2):
    
    # Fetching all data
    reg_data, classification_data = __fetch_all_data()
    print("All data sets have been loaded...")
    __safe_all_data(reg_data, classification_data)
    print("All data sets have been saved as pickle file...")

    # Train all models on each data set
    models = __train_all_model_data_combinations(reg_data,classification_data)
    print("All models are trained...")
    
    # Save all models to pickle
    __safe_all_models(models)
    print("All models have been saved as pickle file...")
    
    # Evaluate all models
    __evaluate_all_models(models)

def __fetch_all_data(test_split: float = 0.2):
    reg_data = []
    classification_data = []
    #reg_data.append(("Abalone", fetch_abalone(normalize=True, test_split=test_split)))
    #reg_data.append(("WineQuality", fetch_wine_quality(normalize=True, test_split=test_split)))
    #reg_data.append(("Diabetes", fetch_diabetes(normalize=True, test_split=test_split)))
    reg_data.append(("Music", fetch_music(normalize=True, test_split=test_split)))
    #classification_data.append(("WineQuality", fetch_wine_quality(normalize=True, test_split=test_split)))
    return reg_data, classification_data

def __safe_all_data(reg_data,classification_data):
    #Get data from data file and save it to a dictionary
    data_dict = {}
    try: 
        file = open("data/all_datasets.pkl", "rb")
        data_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        print("No existing data file found, creating a new one.")
        data_dict = {}
        

    for data_name, (X_train, X_test, y_train, y_test) in reg_data:
        print(f"{data_name}_Regression")
        data_dict[f"{data_name}_Regression"] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
        
    for data_name, (X_train, X_test, y_train, y_test) in classification_data:
        print(f"{data_name}_Classification")
        data_dict[f"{data_name}_Classification"] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
        
    #Save the dictionairy to pickle file
    file = open("data/all_datasets.pkl", "wb")
    pickle.dump(data_dict, file)
    file.close()


def __train_all_model_data_combinations(reg_data,classification_data):
    trained_models = []
    for data_name, (X_train, X_test, y_train, y_test) in reg_data:
        #0=Regression, model_name, data_set, model, X_train, X_test, y_train, y_test
        trained_models.append((0,"LinRegression", data_name, train_LinRegression(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((0,"RFRegressor", data_name, train_RFregressor(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((0,"SVMRegressor", data_name, train_SVMregressor(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((0,"XGBRegressor", data_name, train_XGBregressor(X_train, y_train),X_train, X_test, y_train, y_test))

        
    for data_name, (X_train, X_test, y_train, y_test) in classification_data:
        #1=Classification, model_name, data_set, model, X_train, X_test, y_train, y_test
        trained_models.append((1,"LogRegression", data_name, train_LogRegression(X_train, y_train),X_train, X_test, y_train, y_test))       
        trained_models.append((1,"RFClassifier", data_name, train_RFclassifier(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((1,"SVMClassifier", data_name, train_SVMclassifier(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((1,"XGBClassifier", data_name, train_XGBclassifier(X_train, y_train),X_train, X_test, y_train, y_test))

    return trained_models

def __safe_all_models(models):
    #Open model file and get existing models
    model_dict = {}
    try: 
        file = open("models/all_models.pkl", "rb")
        model_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        print("No existing model file found, creating a new one.")
        model_dict = {}
    
    #Add new models to the dictionary
    for _, model_name, data_set, model, _, _, _, _ in models:
        model_dict[f"{model_name}_{data_set}"] = model

    #Save the dictionairy to pickle file
    file = open("models/all_models.pkl", "wb")
    pickle.dump(model_dict, file)
    file.close()

def __evaluate_all_models(models):
    for indicator, model_name, data_set, model, _, X_test, _, y_test in models:
        if indicator == 0:
            # Regression
            y_hat = model.predict(X_test)
            mse = mean_squared_error(y_test, y_hat)
            print(f"{model_name}_{data_set}, MSE: {mse}")
        else:
            # Classification
            y_hat = model.predict(X_test)
            mse = accuracy_score(y_test, y_hat)
            print(f"{model_name}_{data_set}, Accuracy Score: {mse}")

    return models
  