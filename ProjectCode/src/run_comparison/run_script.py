import subprocess
from pathlib import Path
import itertools
import sys
import os

src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir)) 
os.chdir(str(src_dir)) 

from train import train_models
from evaluations.stability import compute_stability_score
from evaluations.rbo import compute_rbo_score
from evaluations.fidelity import compute_fidelity_score




def __write_result_to_file_and_print(result_content, filename="run_results.txt"):
    print(result_content)
    with open(filename, 'a') as file:
        file.write(f"{result_content}\n")

#Regression Models
regression_models = ["RR", "SR", "XR"] #["LR", "RR", "SR", "XR"]

#Classification Models
classification_models = ["LO", "RC", "SC", "XC"]

#Datasets Regression
regression_data = ["A", "D", "W", "B"]

#Datasets Classification
classification_data = ["W"]

#All Approximation-Methods
approximations = ["K", "S", "C", "G", "E"]

#nsamples values
nsamples_values = [10]#[10, 100, 1000]

#number of runs for stability evaluation
number_of_runs = 10


#To keep track of the number of running processes
running_counter = 0 

#train_models()  # Train all models and save them
__write_result_to_file_and_print("Training Models finished\n")

#All Combinations of datasets, regression models, approximations and nsamples
__write_result_to_file_and_print("Running Process Regression started\n")
for data, model, approximation, nsamples in itertools.product(regression_data, regression_models, approximations, nsamples_values): #iterating over all possible combinations
    if data == "B" and approximation == "E":
        continue
    
    running_counter += 1
    if approximation == "E" and nsamples == nsamples_values[0]: #Exact SHAP does not need nsamples only one run needed
        key = f"D={data}, M={model}, A={approximation}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
        ]
        
    elif approximation == "E":
        continue
       
    else: 
        key = f"D={data}, M={model}, A={approximation}, N={nsamples}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(nsamples)
        ]
    
    try:
        __write_result_to_file_and_print(f"Started: {key} ({running_counter})")
        subprocess.run(running_command, check=True, cwd=src_dir)
        __write_result_to_file_and_print(f"Finished: {key} ({running_counter})")
    except Exception as e:
        __write_result_to_file_and_print(f"Error for: {key} = {str(e)} ({running_counter})")
        
        
    #Stability evaluation for each approximation
    if approximation != "E":
        key_multiple_runs = f"D={data}, M={model}, A={approximation}, N={nsamples}, R={number_of_runs}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(nsamples),
            "--number_of_runs", str(number_of_runs)
        ]
        
        try:
            __write_result_to_file_and_print(f"Started: {key_multiple_runs} ({running_counter})")
            subprocess.run(running_command, check=True, cwd=src_dir)
            __write_result_to_file_and_print(f"Finished: {key_multiple_runs} ({running_counter})")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: {key_multiple_runs} = {str(e)} ({running_counter})")

#All Combinations of datasets, regression models, approximations and nsamples
__write_result_to_file_and_print("Running Process Classification started\n")
for data, model, approximation, nsamples in itertools.product(classification_data, classification_models, approximations, nsamples_values): #iterating over all possible combinations
    running_counter += 1
    if approximation == "E" and nsamples == nsamples_values[0]: #Exact SHAP does not need nsamples only one run needed
        key = f"D={data}, M={model}, A={approximation}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
        ]
    else: 
        key = f"D={data}, M={model}, A={approximation}, N={nsamples}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(nsamples)
        ]
    
    try:
        __write_result_to_file_and_print(f"Started: {key} ({running_counter})")
        subprocess.run(running_command, check=True, cwd=src_dir)
        __write_result_to_file_and_print(f"Finished: {key} ({running_counter})")
    except Exception as e:
        __write_result_to_file_and_print(f"Error for: {key} = {str(e)} ({running_counter})")
    
    #Stability evaluation for each approximation
    if approximation != "E":
        key_multiple_runs = f"D={data}, M={model}, A={approximation}, N={nsamples}, R={number_of_runs}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(nsamples),
            "--number_of_runs", str(number_of_runs)
        ]
        
        try:
            __write_result_to_file_and_print(f"Started: {key_multiple_runs} ({running_counter})")
            subprocess.run(running_command, check=True, cwd=src_dir)
            __write_result_to_file_and_print(f"Finished: {key_multiple_runs} ({running_counter})")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: {key_multiple_runs} = {str(e)} ({running_counter})")
        
__write_result_to_file_and_print("Running Processes ended\n")


__write_result_to_file_and_print("Evalutation Processes Regression started\n")
for data, model, approximation, nsamples in itertools.product(regression_data, regression_models, approximations, nsamples_values): #iterating over all possible combinations
    
    if approximation == "E":
        continue # Skip Exact SHAP for evaluation
    
    fidelity_score = compute_fidelity_score(data, model, approximation, nsamples)
    #fidelity_score = 0
    __write_result_to_file_and_print(f"Computed Fidelity Score for {data}, {model}, {approximation}, {nsamples} = {fidelity_score}")
    
    rbo_score = compute_rbo_score(data, model, approximation, nsamples)
    #rbo_score = 0
    __write_result_to_file_and_print(f"Computed RBO Score for {data}, {model}, {approximation}, {nsamples} = {rbo_score}")
    
    #stability_score = compute_stability_score(data, model, approximation, nsamples, number_of_runs)
    stability_score = 0
    __write_result_to_file_and_print(f"Computed Stability Score for {data}, {model}, {approximation}, {nsamples} = {stability_score}")


__write_result_to_file_and_print("Evalutation Processes Classification started\n")
for data, model, approximation, nsamples in itertools.product(classification_data, classification_models, approximations, nsamples_values): #iterating over all possible combinations
    if approximation == "E":
        continue # Skip Exact SHAP for evaluation
    
    fidelity_score = compute_fidelity_score(data, model, approximation, nsamples)
    #fidelity_score = 0
    __write_result_to_file_and_print(f"Computed Fidelity Score for {data}, {model}, {approximation}, {nsamples} = {fidelity_score}")
    
    rbo_score = compute_rbo_score(data, model, approximation, nsamples)
    #rbo_score = 0
    __write_result_to_file_and_print(f"Computed RBO Score for {data}, {model}, {approximation}, {nsamples} = {rbo_score}")
    
    stability_score = compute_stability_score(data, model, approximation, nsamples, number_of_runs)
    #stability_score = 0
    __write_result_to_file_and_print(f"Computed Stability Score for {data}, {model}, {approximation}, {nsamples} = {stability_score}")
    
    
__write_result_to_file_and_print("Evalutation Processes ended\n")   
