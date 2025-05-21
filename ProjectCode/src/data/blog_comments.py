import pandas as pd
import os
from data.data_splitter import split_data
from data.data_normalizer import normalize_data

def fetch_blog(normalize: bool = True, test_split: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data_folder = "data/blog_comments_data"
    blog_train = pd.read_csv(f"{data_folder}/blogData_train.csv", header=None) #No Column names provided
    blog_test = pd.DataFrame()
    test_file_names = [file_name for file_name in os.listdir(data_folder) if file_name.startswith("blogData_test") and file_name.endswith(".csv")]
    for test_file_name in test_file_names:
       current_test_data = pd.read_csv(f"{data_folder}/{test_file_name}", header=None)
       blog_test = pd.concat([blog_test, current_test_data], ignore_index=True)

    blog_X_train = blog_train.iloc[:, :-1]
    blog_y_train = blog_train.iloc[:, -1]
    blog_X_test = blog_test.iloc[:, :-1]
    blog_y_test = blog_test.iloc[:, -1]
    

    # Identify categorical columns based on the criteria thtey have less than 3 unique values and have the binary bounds
    list_max = [maxima for maxima in blog_X_train.max()]
    list_min = [minima for minima in blog_X_train.min()]
    list_unique = [unique_count for unique_count in blog_X_train.nunique()]
    cat_columns = []
    for index in range(blog_X_train.shape[1]):
        if list_unique[index] < 3 and list_min[index] == 0 and list_max[index] == 1:
            cat_columns.append(blog_X_train.columns[index])
    
    if normalize:
        blog_X_train, blog_X_test = normalize_data(blog_X_train, blog_X_test, cat_columns)
        
    return blog_X_train, blog_X_test, blog_y_train, blog_y_test