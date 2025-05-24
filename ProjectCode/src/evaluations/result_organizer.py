import pickle

def save_result(data_name, model_name, approximation_name, nsamples, category, result, number_of_runs = None):
    file_name = f"evaluations/results.pkl"
    if number_of_runs is not None:
        key = f"{approximation_name}_{data_name}_{model_name}_{nsamples}_{number_of_runs}"
    else:
        key = f"{approximation_name}_{data_name}_{model_name}_{nsamples}"
    try:
        file = open(file_name, "rb")
        evaluation_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        evaluation_dict = dict()

    if key not in evaluation_dict:
        evaluation_dict[key] = dict() 
    evaluation_dict[key][category] = result

    
    file = open(file_name, "wb") 
    pickle.dump(evaluation_dict, file)
    file.close()
    
def print_results():
    file_name = f"evaluations/results.pkl"
    try:
        file = open(file_name, "rb")
        evaluation_dict = pickle.load(file)
        file.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_name} not found")
    for key, value in evaluation_dict.items():
        print("\n"+key)
        for evaluation_key, result_value in value.items():
            print(f"    - {evaluation_key}: {result_value}")