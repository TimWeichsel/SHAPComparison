import pickle
import pandas as pd

#fetch data functions imports
from data.abalone import fetch_abalone
from data.wine_quality import fetch_wine_quality
from data.diabetes import fetch_diabetes
from data.music import fetch_music

#model functions imports
from models.regression import train_LinRegression
from models.random_forest import train_RFregressor
from models.support_vector_machine import train_SVMregressor
from models.xgboost import train_XGBregressor

#model evaluation imports
from sklearn.metrics import mean_squared_error

def train_models(test_split: float = 0.2) -> None:
    '''
    Loading and saving all data sets, train all models on all data sets, saving and printing evaluation of all models
    
    Parameters:
        test_split -- Fraction of data that is used for the test set (default: 0.2 = 20% test data)
    '''
    # Fetching all data
    reg_data = __fetch_all_data(test_split)
    print("All data sets have been loaded...")
    __safe_all_data(reg_data)
    print("All data sets have been saved as pickle file...")

    # Train all models on each data set
    models = __train_all_model_data_combinations(reg_data)
    print("All models are trained...")
    
    # Save all models to pickle
    __safe_all_models(models)
    print("All models have been saved as pickle file...")
    
    # Evaluate all models
    __evaluate_all_models(models)

def __fetch_all_data(test_split: float = 0.2) -> list[tuple[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]]:
    '''
    Fetch all datasets, normmalize (and one hot encode) them and split them into training/test sets
    
    Parameters:
        test_split -- Fraction of data that is used for the test set (default: 0.2 = 20% test data)

    Returns:
        List list of tuples with name of data set and tuple of DataFrames/Series for training/test features/targets
    
    '''
    reg_data = []
    reg_data.append(("Abalone", fetch_abalone(normalize=True, test_split=test_split)))
    reg_data.append(("WineQuality", fetch_wine_quality(normalize=True, test_split=test_split)))
    reg_data.append(("Diabetes", fetch_diabetes(normalize=True, test_split=test_split)))
    reg_data.append(("Music", fetch_music(normalize=True, test_split=test_split)))
    return reg_data

def __safe_all_data(reg_data) -> None:
    '''
    Add the new data to the data file "data/all_datasets.pkl" and save it back.
    
    Parameters:
        reg_data -- Data to be stored in the pickle file
    
    '''
    
    data_dict = {}
    try: 
        file = open("data/all_datasets.pkl", "rb")
        data_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        print("No existing data file found, creating a new one.")
        data_dict = {}
        

    for data_name, (X_train, X_test, y_train, y_test) in reg_data:
        data_dict[f"{data_name}_Regression"] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
        
    #Save the dictionairy to pickle file
    file = open("data/all_datasets.pkl", "wb")
    pickle.dump(data_dict, file)
    file.close()


def __train_all_model_data_combinations(reg_data: list) -> list:
    '''
    Train for each dataset passed with reg_data, all specified ML models
    
    Parameters:
        reg_data -- Data to be trained on the specified ML models (type: list[tuple[<Name>, tuple[X_train, X_test, y_train, y_test]]])
    
    Returns:
        List of all trained models
    '''
    trained_models = []
    for data_name, (X_train, X_test, y_train, y_test) in reg_data:
        #0=Regression, model_name, data_set, model, X_train, X_test, y_train, y_test
        trained_models.append((0,"LinRegression", data_name, train_LinRegression(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((0,"RFRegressor", data_name, train_RFregressor(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((0,"SVMRegressor", data_name, train_SVMregressor(X_train, y_train),X_train, X_test, y_train, y_test))
        trained_models.append((0,"XGBRegressor", data_name, train_XGBregressor(X_train, y_train),X_train, X_test, y_train, y_test))
    #classification models are not implemented
    return trained_models

def __safe_all_models(models: list) -> None:
    '''
    Safe all models passed to models/all_models.pkl file
    
    Parameters:
        models -- ML modles to be saved to pickle file (type: list[tuple[0=Regression, model_name, data_set, model, X_train, X_test, y_train, y_test]]])
    '''
    #Open model file and save existing models to dictionary
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

    #Save the updated dictionairy to pickle file
    file = open("models/all_models.pkl", "wb")
    pickle.dump(model_dict, file)
    file.close()

def __evaluate_all_models(models: list) -> None:
    '''
    Evaluate (MSE) all models passed on the test set and print the results
    
    Parameters:
        models -- ML modles to be saved to pickle file (type: list[tuple[0=Regression, model_name, data_set, model, X_train, X_test, y_train, y_test]]])
    '''
    for indicator, model_name, data_set, model, _, X_test, _, y_test in models:
        if indicator == 0:
            # Regression
            y_hat = model.predict(X_test)
            mse = mean_squared_error(y_test, y_hat)
            print(f"{model_name}_{data_set}, MSE: {mse}")
        else:
            raise ValueError("Classification models are not implemented")

  