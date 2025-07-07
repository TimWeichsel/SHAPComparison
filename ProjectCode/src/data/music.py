import pandas as pd
from data.data_splitter import split_data
from data.data_normalizer import normalize_data

def fetch_music(normalize: bool = True, test_split: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data_folder = "data/music_origin_data"
    feature_names = [f"feature_{i}" for i in range(0, 67)] + ["latitude"] + ["longitude"]  #68 audio features of the track + latitude & longitude of the origin
    
    music = pd.read_csv(f"{data_folder}/default_features_1059_tracks.txt",sep=",", header=None, names=feature_names) #Infer Column names
    music_X = music.iloc[:, :15] #only considering the first audio features as predictors
    music_y = music["longitude"] #predict longitude based on the audio features
    
    #Split the data into train and test sets
    music_X_train, music_X_test, music_y_train, music_y_test = split_data(music_X, music_y, test_split=test_split)
    
    # Identify categorical columns based on the criteria that they have less than 3 unique values and have the binary bounds - should be no categorical columns in this dataset
    list_max = [maxima for maxima in music_X_train.max()]
    list_min = [minima for minima in music_X_train.min()]
    list_unique = [unique_count for unique_count in music_X_train.nunique()]
    cat_columns = []
    for index in range(music_X_train.shape[1]):
        if list_unique[index] < 3 and list_min[index] == 0 and list_max[index] == 1:
            raise ValueError(f"Categorical column found {index}: ", music_X_train.columns[index])
    
    if normalize:
        music_X_train, music_X_test = normalize_data(music_X_train, music_X_test, cat_columns)
        
    return music_X_train, music_X_test, music_y_train, music_y_test
