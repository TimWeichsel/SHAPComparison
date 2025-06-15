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
regression_models = ["LR", "RR", "SR", "XR"] #["LR", "RR", "SR", "XR"]

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

def __run_all():
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


def run_all_reg_sample_specific_nsamples(n_samples):
    print("Running all regression models with nsamples={n_samples}")
    running_counter = 0
    for data, model in itertools.product(regression_data, regression_models): #iterating over all possible combinations
        running_counter += 1

        key = f"D={data}, M={model}, A=S, N={n_samples}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", "S",
            "--nsamples", str(n_samples)]
        
        try:
            __write_result_to_file_and_print(f"Started: {key} ({running_counter})")
            subprocess.run(running_command, check=True, cwd=src_dir)
            __write_result_to_file_and_print(f"Finished: {key} ({running_counter})")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: {key} = {str(e)} ({running_counter})")
            
            
def run_all_wine_runtime_matched():
    print("Running all wine-models with runtime-matched nsamples.")
    running_counter = 0
    src = str(src_dir)

    run_queue = [
    ('C', 'SR', 25),
    ('C', 'XR', 75),
    ('G', 'LR', 1000000000),
    ('G', 'RR', 10000),   
    ('G', 'SR', 20),
    #('G', 'RR', 1),
    #('G', 'XR', 7),
    ('S', 'RR', 20)]
    #('S', 'XR', 3200),
    #('K', 'RR', 1)]
        
    '''
    # ≈10s Kategorie
    ('C', 'LR', 5),
    # ≈100s Kategorie
    ('C', 'LR', 60),
    # ≈250s Kategorie
    ('C', 'LR', 5000),
    # ≈500s Kategorie
    ('C', 'LR', 10000),
    # ≈750s Kategorie
    ('C', 'LR', 15000),
    # ≈1000s Kategorie
    ('C', 'LR', 30000),'''
        
    '''
    # ≈10s Kategorie
    # S_W_LR_30 + Laufzeit: 7.79s
    # S_W_RR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # S_W_SR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # S_W_XR_30 + Laufzeit: 7.2s
    # K_W_LR_15 + Laufzeit: 11.9s
    # K_W_RR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # K_W_SR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # K_W_XR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # C_W_LR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # C_W_RR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # C_W_SR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # C_W_XR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # G_W_LR_10 + Laufzeit: 7.73s
    # G_W_RR_20 + Laufzeit: 12.81s
    # G_W_SR → Laufzeit bei kleinstem n_samples bereits > 15s, keine passend kleinere n_samples möglich
    # G_W_XR_1 + Laufzeit: 8.08s
    
    # ≈100s Kategorie
    # S_W_LR_4000 + Laufzeit: 103.66s
    # S_W_RR → Laufzeit bei kleinstem n_samples bereits > 120s, keine passend kleinere n_samples möglich
    # S_W_SR_50 + Laufzeit: 83.82s
    # S_W_XR + geschätzt: n_samples ≈ 2810 für ca. ≈100s (aus Werten: [30, 50, 80, 120, 150, 250, 400, 750] → [7.2, 8.4, 9.0, 9.7, 14.9, 13.6, 16.4, 25.9])
    #('S', 'XR', 2810),
    # K_W_LR + geschätzt: n_samples ≈ 200 für ca. ≈100s (aus Werten: [15, 20, 40, 100, 400, 750, 1000] → [11.9, 21.1, 21.3, 48.4, 276.1, 534.5, 661.9])
    #('K', 'LR', 200),
    # K_W_RR → Laufzeit bei kleinstem n_samples bereits > 120s, keine passend kleinere n_samples möglich
    # K_W_SR → Laufzeit bei kleinstem n_samples bereits > 120s, keine passend kleinere n_samples möglich
    # K_W_XR + geschätzt: n_samples ≈ 5 für ca. ≈100s (aus Werten: [1, 2, 10, 20, 40] → [20.6, 34.8, 138.7, 265.5, 521.3])
    #('K', 'XR', 6),
    # C_W_LR + geschätzt: n_samples ≈ 75 für ca. ≈100s (aus Werten: [10, 40, 100, 750, 1000] → [17.3, 58.3, 147.6, 293.7, 293.7])
    #('C', 'LR', 80),
    # C_W_RR + geschätzt: n_samples ≈ 70 für ca. ≈100s (aus Werten: [1, 10, 20, 100, 750] → [19.1, 17.1, 32.1, 151.0, 330.8])
    #('C', 'RR', 70),
    # C_W_SR + geschätzt: n_samples ≈ 20 für ca. ≈100s (aus Werten: [1, 5, 10, 100] → [65.0, 59.9, 62.8, 734.9])
    #('C', 'SR', 20), -> kleiner geht nicht!
    # C_W_XR + geschätzt: n_samples ≈ 70 für ca. ≈100s (aus Werten: [1, 2, 10, 20, 40] → [19.1, 17.6, 16.1, 31.1, 59.3])
    #('C', 'XR', 70),
    # G_W_LR + geschätzt: n_samples ≈ 1630 für ca. ≈100s (aus Werten: [10, 40, 100, 400, 750, 1000] → [7.7, 17.6, 35.0, 59.5, 72.0, 60.9])
    #('G', 'LR', 1630),
    # G_W_RR_750 + Laufzeit: 89.24s
    # G_W_SR + geschätzt: n_samples ≈ 10 für ca. ≈100s (aus Werten: [2, 5, 10, 100] → [56.6, 54.4, 52.9, 626.6])
    #('G', 'SR', 20),
    # G_W_XR + geschätzt: n_samples ≈ 230 für ca. ≈100s (aus Werten: [1, 2, 10, 20, 40] → [8.1, 8.4, 8.1, 11.0, 17.0])
    #('G', 'XR', 230),
    
    # ≈250s Kategorie
    # S_W_LR + geschätzt: n_samples ≈ 9600 für ca. ≈250s (aus Werten: [30, 40, 50, 80, 100, 150, 400, 750, 1000, 4000] → [7.8, 11.9, 7.1, 11.0, 10.9, 10.8, 16.1, 25.0, 31.6, 103.7])
    #('S', 'LR', 9600),
    # S_W_RR + geschätzt: n_samples ≈ 2240 für ca. ≈250s (aus Werten: [30, 50, 80, 100, 150, 1000, 4000] → [129.5, 160.7, 129.6, 164.5, 160.0, 189.1, 448.7])
    ('S', 'RR', 1800),
    # S_W_SR + geschätzt: n_samples ≈ 200 für ca. ≈250s (aus Werten: [30, 50, 80, 150, 400] → [49.1, 83.8, 105.7, 199.8, 528.1])
    #('S', 'SR', 200),
    # S_W_XR + geschätzt: n_samples ≈ 6980 für ca. ≈250s (aus Werten: [30, 50, 80, 120, 150, 250, 400, 750] → [7.2, 8.4, 9.0, 9.7, 14.9, 13.6, 16.4, 25.9])
    ('S', 'XR', 8000),
    # K_W_LR_400 + Laufzeit: 276.11s
    # K_W_RR → Laufzeit bei kleinstem n_samples bereits > 300s, keine passend kleinere n_samples möglich
    # K_W_SR → Laufzeit bei kleinstem n_samples bereits > 300s, keine passend kleinere n_samples möglich
    # K_W_XR_20 + Laufzeit: 265.47s
    # C_W_LR_750 + Laufzeit: 293.73s
    # C_W_RR + geschätzt: n_samples ≈ 560 für ca. ≈250s (aus Werten: [1, 10, 20, 100, 750] → [19.1, 17.1, 32.1, 151.0, 330.8])
    ('C', 'RR', 450),
    # C_W_SR + geschätzt: n_samples ≈ 30 für ca. ≈250s (aus Werten: [1, 5, 10, 100] → [65.0, 59.9, 62.8, 734.9])
    #('C', 'SR', 20),
    # C_W_XR + geschätzt: n_samples ≈ 160 für ca. ≈250s (aus Werten: [1, 2, 10, 20, 40] → [19.1, 17.6, 16.1, 31.1, 59.3])
    #('C', 'XR', 150),
    # G_W_LR + geschätzt: n_samples ≈ 4070 für ca. ≈250s (aus Werten: [10, 40, 100, 400, 750, 1000] → [7.7, 17.6, 35.0, 59.5, 72.0, 60.9])
    ('G', 'LR', 13000),
    # G_W_RR + geschätzt: n_samples ≈ 2090 für ca. ≈250s (aus Werten: [2, 10, 20, 40, 100, 750] → [9.4, 8.9, 12.8, 19.8, 45.8, 89.2])
    ('G', 'RR', 5000),
    # G_W_SR + geschätzt: n_samples ≈ 45 für ca. ≈250s (aus Werten: [2, 5, 10, 100] → [56.6, 54.4, 52.9, 626.6])
    ('G', 'SR', 50),
    # G_W_XR + geschätzt: n_samples ≈ 570 für ca. ≈250s (aus Werten: [1, 2, 10, 20, 40] → [8.1, 8.4, 8.1, 11.0, 17.0])
    ('G', 'XR', 2600),
    
    # ≈500s Kategorie
    # S_W_LR + geschätzt: n_samples ≈ 19170 für ca. ≈500s (aus Werten: [30, 40, 50, 80, 100, 150, 400, 750, 1000, 4000] → [7.8, 11.9, 7.1, 11.0, 10.9, 10.8, 16.1, 25.0, 31.6, 103.7])
    #('S', 'LR', 19170),
    # S_W_RR_4000 + Laufzeit: 448.7s
    # S_W_SR_400 + Laufzeit: 528.14s
    # S_W_XR + geschätzt: n_samples ≈ 13940 für ca. ≈500s (aus Werten: [30, 50, 80, 120, 150, 250, 400, 750] → [7.2, 8.4, 9.0, 9.7, 14.9, 13.6, 16.4, 25.9])
    ('S', 'XR', 18000),
    # K_W_LR_750 + Laufzeit: 534.55s
    # K_W_RR_10 + Laufzeit: 470.88s
    # K_W_SR → Laufzeit bei kleinstem n_samples bereits > 600s, keine passend kleinere n_samples möglich
    # K_W_XR_40 + Laufzeit: 521.34s
    # C_W_LR + geschätzt: n_samples ≈ 1690 für ca. ≈500s (aus Werten: [10, 40, 100, 750, 1000] → [17.3, 58.3, 147.6, 293.7, 293.7])
    ('C', 'LR', 4000),
    # C_W_RR + geschätzt: n_samples ≈ 1130 für ca. ≈500s (aus Werten: [1, 10, 20, 100, 750] → [19.1, 17.1, 32.1, 151.0, 330.8])
    ('C', 'RR', 3000),
    # C_W_SR + geschätzt: n_samples ≈ 60 für ca. ≈500s (aus Werten: [1, 5, 10, 100] → [65.0, 59.9, 62.8, 734.9])
    ('C', 'SR', 60),
    # C_W_XR + geschätzt: n_samples ≈ 320 für ca. ≈500s (aus Werten: [1, 2, 10, 20, 40] → [19.1, 17.6, 16.1, 31.1, 59.3])
    ('C', 'XR', 700),
    # G_W_LR + geschätzt: n_samples ≈ 8140 für ca. ≈500s (aus Werten: [10, 40, 100, 400, 750, 1000] → [7.7, 17.6, 35.0, 59.5, 72.0, 60.9])
    ('G', 'LR', 100000),
    # G_W_RR + geschätzt: n_samples ≈ 4190 für ca. ≈500s (aus Werten: [2, 10, 20, 40, 100, 750] → [9.4, 8.9, 12.8, 19.8, 45.8, 89.2])
    ('G', 'RR', 33000),
    # G_W_SR + geschätzt: n_samples ≈ 80 für ca. ≈500s (aus Werten: [2, 5, 10, 100] → [56.6, 54.4, 52.9, 626.6])
    ('G', 'SR', 80),
    # G_W_XR + geschätzt: n_samples ≈ 1140 für ca. ≈500s (aus Werten: [1, 2, 10, 20, 40] → [8.1, 8.4, 8.1, 11.0, 17.0])
    ('G', 'XR', 7000),
    
    # ≈750s Kategorie
    # S_W_LR + geschätzt: n_samples ≈ 28750 für ca. ≈750s (aus Werten: [30, 40, 50, 80, 100, 150, 400, 750, 1000, 4000] → [7.8, 11.9, 7.1, 11.0, 10.9, 10.8, 16.1, 25.0, 31.6, 103.7])

    ('S', 'LR', 33000),
    # S_W_RR + geschätzt: n_samples ≈ 6660 für ca. ≈750s (aus Werten: [30, 50, 80, 100, 150, 1000, 4000] → [129.5, 160.7, 129.6, 164.5, 160.0, 189.1, 448.7])
    ('S', 'RR', 9500),
    # S_W_SR + geschätzt: n_samples ≈ 550 für ca. ≈750s (aus Werten: [30, 50, 80, 150, 400] → [49.1, 83.8, 105.7, 199.8, 528.1])
    ('S', 'SR', 550),
    # S_W_XR + geschätzt: n_samples ≈ 20900 für ca. ≈750s (aus Werten: [30, 50, 80, 120, 150, 250, 400, 750] → [7.2, 8.4, 9.0, 9.7, 14.9, 13.6, 16.4, 25.9])
    ('S', 'XR', 28000),
    # K_W_LR_1000 + Laufzeit: 661.92s
    # K_W_RR + geschätzt: n_samples ≈ 20 für ca. ≈750s (aus Werten: [10, 20, 40, 100] → [470.9, 917.8, 1687.9, 4417.1])
    ('K', 'RR', 20),
    # K_W_SR → Laufzeit bei kleinstem n_samples bereits > 850s, keine passend kleinere n_samples möglich
    # K_W_XR + geschätzt: n_samples ≈ 50 für ca. ≈750s (aus Werten: [1, 2, 10, 20, 40] → [20.6, 34.8, 138.7, 265.5, 521.3])
    ('K', 'XR', 60),
    # C_W_LR + geschätzt: n_samples ≈ 2530 für ca. ≈750s (aus Werten: [10, 40, 100, 750, 1000] → [17.3, 58.3, 147.6, 293.7, 293.7])
    ('C', 'LR', 14000),
    # C_W_RR + geschätzt: n_samples ≈ 1690 für ca. ≈750s (aus Werten: [1, 10, 20, 100, 750] → [19.1, 17.1, 32.1, 151.0, 330.8])
    ('C', 'RR', 15000),
    # C_W_SR_100 + Laufzeit: 734.9s
    # C_W_XR + geschätzt: n_samples ≈ 490 für ca. ≈750s (aus Werten: [1, 2, 10, 20, 40] → [19.1, 17.6, 16.1, 31.1, 59.3])
    ('C', 'XR', 3300),
    # G_W_LR + geschätzt: n_samples ≈ 12210 für ca. ≈750s (aus Werten: [10, 40, 100, 400, 750, 1000] → [7.7, 17.6, 35.0, 59.5, 72.0, 60.9])
    ('G', 'LR', 1800000),
    # G_W_RR + geschätzt: n_samples ≈ 6280 für ca. ≈750s (aus Werten: [2, 10, 20, 40, 100, 750] → [9.4, 8.9, 12.8, 19.8, 45.8, 89.2])
    ('G', 'RR', 1000000),
    # G_W_SR + geschätzt: n_samples ≈ 110 für ca. ≈750s (aus Werten: [2, 5, 10, 100] → [56.6, 54.4, 52.9, 626.6])
    #('G', 'SR', 110),
    # G_W_XR + geschätzt: n_samples ≈ 1710 für ca. ≈750s (aus Werten: [1, 2, 10, 20, 40] → [8.1, 8.4, 8.1, 11.0, 17.0])
    ('G', 'XR', 20000),
    
    # ≈1000s Kategorie
    # S_W_LR + geschätzt: n_samples ≈ 38320 für ca. ≈1000s (aus Werten: [30, 40, 50, 80, 100, 150, 400, 750, 1000, 4000] → [7.8, 11.9, 7.1, 11.0, 10.9, 10.8, 16.1, 25.0, 31.6, 103.7])
    ('S', 'LR', 50000),
    # S_W_RR + geschätzt: n_samples ≈ 8870 für ca. ≈1000s (aus Werten: [30, 50, 80, 100, 150, 1000, 4000] → [129.5, 160.7, 129.6, 164.5, 160.0, 189.1, 448.7])
    ('S', 'RR', 13000),
    # S_W_SR + geschätzt: n_samples ≈ 730 für ca. ≈1000s (aus Werten: [30, 50, 80, 150, 400] → [49.1, 83.8, 105.7, 199.8, 528.1])
    ('S', 'SR', 900),
    # S_W_XR + geschätzt: n_samples ≈ 27850 für ca. ≈1000s (aus Werten: [30, 50, 80, 120, 150, 250, 400, 750] → [7.2, 8.4, 9.0, 9.7, 14.9, 13.6, 16.4, 25.9])
    ('S', 'XR', 40000),
    # K_W_LR + geschätzt: n_samples ≈ 1500 für ca. ≈1000s (aus Werten: [15, 20, 40, 100, 400, 750, 1000] → [11.9, 21.1, 21.3, 48.4, 276.1, 534.5, 661.9])
    ('K', 'LR', 1500),
    # K_W_RR_20 + Laufzeit: 917.79s
    # K_W_SR → Laufzeit bei kleinstem n_samples bereits > 1100s, keine passend kleinere n_samples möglich
    ('K', 'SR', 1),
    # K_W_XR + geschätzt: n_samples ≈ 70 für ca. ≈1000s (aus Werten: [1, 2, 10, 20, 40] → [20.6, 34.8, 138.7, 265.5, 521.3])
    ('K', 'XR', 70),
    # C_W_LR + geschätzt: n_samples ≈ 3380 für ca. ≈1000s (aus Werten: [10, 40, 100, 750, 1000] → [17.3, 58.3, 147.6, 293.7, 293.7])
    ('C', 'LR', 111000),
    # C_W_RR + geschätzt: n_samples ≈ 2260 für ca. ≈1000s (aus Werten: [1, 10, 20, 100, 750] → [19.1, 17.1, 32.1, 151.0, 330.8])
    ('C', 'RR', 6800),
    # C_W_SR + geschätzt: n_samples ≈ 130 für ca. ≈1000s (aus Werten: [1, 5, 10, 100] → [65.0, 59.9, 62.8, 734.9])
    ('C', 'SR', 130),
    # C_W_XR + geschätzt: n_samples ≈ 650 für ca. ≈1000s (aus Werten: [1, 2, 10, 20, 40] → [19.1, 17.6, 16.1, 31.1, 59.3])
    ('C', 'XR', 2000),
    # G_W_LR + geschätzt: n_samples ≈ 16270 für ca. ≈1000s (aus Werten: [10, 40, 100, 400, 750, 1000] → [7.7, 17.6, 35.0, 59.5, 72.0, 60.9])
    ('G', 'LR', 300000),
    # G_W_RR + geschätzt: n_samples ≈ 8380 für ca. ≈1000s (aus Werten: [2, 10, 20, 40, 100, 750] → [9.4, 8.9, 12.8, 19.8, 45.8, 89.2])
    ('G', 'RR', 9000),
    # G_W_SR + geschätzt: n_samples ≈ 150 für ca. ≈1000s (aus Werten: [2, 5, 10, 100] → [56.6, 54.4, 52.9, 626.6])
    #('G', 'SR', 150),
    # G_W_XR + geschätzt: n_samples ≈ 2290 für ca. ≈1000s (aus Werten: [1, 2, 10, 20, 40] → [8.1, 8.4, 8.1, 11.0, 17.0])
    ('G', 'XR', 22900)'''

    
    for approximation, model, n_samples in run_queue:
        running_counter += 1
        key = f"D=W, M={model}, A={approximation}, N={n_samples}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", "W",
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(n_samples)
        ]

        try:
            __write_result_to_file_and_print(f"Started: {key} ({running_counter})")
            subprocess.run(running_command, check=True, cwd=src)
            __write_result_to_file_and_print(f"Finished: {key} ({running_counter})")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: {key} = {str(e)} ({running_counter})")


def run_all_wine_stability():
    
    running_counter = 0
    src = str(src_dir)

    # (approximation, model, n_samples) – sortiert nach geschätzter Runtime (schnelle zuerst)
    run_queue = [
    ('C', 'SR', 5, 10),
    ('C', 'XR', 65, 10),
    ('C', 'RR', 0, 10),
    ('C', 'XR', 0, 10),
    ('G', 'LR', 14000000, 10),
    ('G', 'RR', 62000, 10),  
    ('G', 'SR', 15, 10),
    ('G', 'RR', 1, 10),
    ('G', 'XR', 7, 10),
    ('S', 'RR', 20, 10),
    ('S', 'XR', 3200, 10),
    ('K', 'RR', 1, 10),
        
    ]
    for approximation, model, n_samples, number_of_runs in run_queue:
        running_counter += 1
        key = f"D=W, M={model}, A={approximation}, N={n_samples}, R={number_of_runs}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", "W",
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(n_samples),
            "--number_of_runs", str(number_of_runs)
        ]

        try:
            __write_result_to_file_and_print(f"Started: {key} ({running_counter})")
            subprocess.run(running_command, check=True, cwd=src)
            __write_result_to_file_and_print(f"Finished: {key} ({running_counter})")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: {key} = {str(e)} ({running_counter})")
    
    
def run_exact_diabetes():
    
    running_counter = 0
    src = str(src_dir)

    # (approximation, model, n_samples) – sortiert nach geschätzter Runtime (schnelle zuerst)
    run_queue = [
'''
    ('C', 'LR',  'D', '250000'),
    ('C', 'LR',  'D', '6000000'),
    ('C', 'LR',  'D', '80000000'),
    ('C', 'LR',  'D', '30000000'),
    ('C', 'LR',  'D', '70000000'),
    ('C', 'LR',  'D', '110000000'),
    ('C', 'LR',  'D', '150000000'),
    
    ('C', 'RR',  'D', '300'),
    ('C', 'RR',  'D', '50000000'),
    ('C', 'RR',  'D', '3000000'),
    ('C', 'RR',  'D', '500000000'),
    ('C', 'RR',  'D', '800000000'),
    ('C', 'RR',  'D', '1600000000'),
    ('C', 'RR',  'D', '2000000000'),
    
    #('C', 'SR',  'D', '120'),
    ('C', 'SR',  'D', '4000000'),
    ('C', 'SR',  'D', '15000000'),
    ('C', 'SR',  'D', '40000000'),
    ('C', 'SR',  'D', '65000000'),
    ('C', 'SR',  'D', '90000000'),
    ('C', 'SR',  'D', '18000000')

    ('C', 'XR',  'D', '5000'),
    ('C', 'XR',  'D', '70000'),
    ('C', 'XR',  'D', '160000'),
    ('C', 'XR',  'D', '400000'),
    ('C', 'XR',  'D', '700000'),
    ('C', 'XR',  'D', '1000000'),
    
    ('C', 'LR',  'W', '3'),
    ('C', 'LR',  'W', '70'),
    ('C', 'LR',  'W', '750'),
    ('C', 'LR',  'W', '1000'),
    ('C', 'LR',  'W', '1690'),
    ('C', 'LR',  'W', '2530'),
    ('C', 'LR',  'W', '3380'),
    ('C', 'LR',  'W', '5000'),
    ('C', 'LR',  'W', '6000'),
    ('C', 'LR',  'W', '20000'),
    ('C', 'LR',  'W', '30000'),
    ('C', 'LR',  'W', '40000'),
    ('C', 'LR',  'W', '50000'),
    ('C', 'LR',  'W', '70000'),
    ('C', 'LR',  'W', '100000'),
    ('C', 'RR',  'W', '12'),
    ('C', 'RR',  'W', '13'),
    ('C', 'RR',  'W', '65'),
    ('C', 'RR',  'W', '560'),
    ('C', 'RR',  'W', '750'),
    ('C', 'RR',  'W', '1690'),
    ('C', 'RR',  'W', '2260'),
    ('C', 'RR',  'W', '7000'),
    ('C', 'RR',  'W', '10000'),    
    ('C', 'RR',  'W', '15000'),
    ('C', 'RR',  'W', '20000'),
    ('C', 'RR',  'W', '30000'),
    ('C', 'RR',  'W', '35000')
      
    ('C', 'SR',  'W', '12'),
    ('C', 'SR',  'W', '13'),
    ('C', 'SR',  'W', '16'),
    ('C', 'SR',  'W', '40'),
    ('C', 'SR',  'W', '80'),
    ('C', 'SR',  'W', '110'),
    ('C', 'SR',  'W', '160'),
    ('C', 'SR',  'W', '180000'),

    ('C', 'XR',  'W', '12'),
    ('C', 'XR',  'W', '13'),
    ('C', 'XR',  'W', '60'),
    ('C', 'XR',  'W', '50'),
    ('C', 'XR',  'W', '45'),
    ('C', 'XR',  'W', '1000'),
    ('C', 'XR',  'W', '490'),
    ('C', 'XR',  'W', '3000'),
    ('C', 'XR',  'W', '5000'),
    ('C', 'XR',  'W', '8000'),
    ('C', 'XR',  'W', '10000')    '''
    ('K', 'SR', '100'),
    ('K', 'LR', '800'),
    ('K', 'XR', '150'),
    ('K', 'RR', '40'),
    
    ('S', 'SR', '100'),
    ('S', 'LR', '8000'),
    ('S', 'XR', '7000'),
    ('S', 'RR', '100'),
    
    ('G', 'SR', '150'),
    ('G', 'LR', '400'),
    ('G', 'XR', '400'),
    ('G', 'RR', '350')

    ]
    for approximation, model, data, n_samples, in run_queue:
        running_counter += 1
        key = f"D=D, M={model}, A={approximation}, D={data}, N={n_samples}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(n_samples)
        ]

        try:
            __write_result_to_file_and_print(f"Started: {key} ({running_counter})")
            subprocess.run(running_command, check=True, cwd=src)
            __write_result_to_file_and_print(f"Finished: {key} ({running_counter})")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: {key} = {str(e)} ({running_counter})")
    
    
    
    