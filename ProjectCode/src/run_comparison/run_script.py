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
    

def run_approximations():
    
    running_counter = 0
    src = str(src_dir)
    
    '''('S', 'LR',  'D', '100'),
    ('S', 'LR',  'D', '200'),
    ('S', 'LR',  'D', '300'),
    ('S', 'LR',  'D', '400'),
    ('S', 'LR',  'D', '500'),
    ('S', 'LR',  'D', '600'),
    ('S', 'LR',  'D', '700'),
    ('S', 'LR',  'D', '800'),
    ('S', 'LR',  'D', '900'),
    ('S', 'LR',  'D', '1000'),
    ('S', 'LR',  'D', '1500'),
    ('S', 'LR',  'D', '2000'),
    ('S', 'LR',  'D', '2500'),
    ('S', 'LR',  'D', '3000'),too long
    ('S', 'LR',  'D', '3500'),#too long
    ('S', 'LR',  'D', '4000'),#too long
    
    ('S', 'RR',  'D', '30'),
    ('S', 'RR',  'D', '50'),
    ('S', 'RR',  'D', '100'),
    ('S', 'RR',  'D', '200'),
    ('S', 'RR',  'D', '300'),
    ('S', 'RR',  'D', '400'),
    ('S', 'RR',  'D', '500'),
    ('S', 'RR',  'D', '600'),
    ('S', 'RR',  'D', '700'),
    ('S', 'RR',  'D', '800'),
    ('S', 'RR',  'D', '900'),
    ('S', 'RR',  'D', '1000'),
    ('S', 'RR',  'D', '1500'),
    ('S', 'RR',  'D', '2000'),
    ('S', 'RR',  'D', '2500'),
    ('S', 'RR',  'D', '3000'),
    ('S', 'RR',  'D', '3500'),
    ('S', 'RR',  'D', '4000'),
    ('S', 'RR',  'D', '4500'),
    ('S', 'RR',  'D', '5000'),
    ('S', 'RR',  'D', '5500'),
    ('S', 'RR',  'D', '6000')
    ('S', 'RR',  'D', '1150'),
    ('S', 'RR',  'D', '1300'),
    ('S', 'RR',  'D', '7000'),
    ('S', 'RR',  'D', '8000'),
    ('S', 'RR',  'D', '9000'),
    ('S', 'RR',  'D', '10000'),
    ('S', 'RR',  'D', '12000'),
    ('S', 'RR',  'D', '14000'),
    ('S', 'RR',  'D', '16000'),
    
    ('S', 'SR',  'D', '30'),
    ('S', 'SR',  'D', '50'),
    ('S', 'SR',  'D', '100'),
    ('S', 'SR',  'D', '150'),
    ('S', 'SR',  'D', '200'),
    ('S', 'SR',  'D', '250'),
    ('S', 'SR',  'D', '300'),
    ('S', 'SR',  'D', '350'),
    ('S', 'SR',  'D', '400'),
    ('S', 'SR',  'D', '450'),
    ('S', 'SR',  'D', '500'),
    ('S', 'SR',  'D', '550'),
    ('S', 'SR',  'D', '600'),
    ('S', 'SR',  'D', '650'),
    ('S', 'SR',  'D', '700'),
    ('S', 'SR',  'D', '750'),
    ('S', 'SR',  'D', '800'),
    ('S', 'SR',  'D', '850'),
    ('S', 'SR',  'D', '1000'),
    ('S', 'SR',  'D', '3000'),
    ('S', 'SR',  'D', '5000'),
    ('S', 'SR',  'D', '10000'),
    ('S', 'SR',  'D', '15000'),
    ('S', 'SR',  'D', '20000'),
    ('S', 'SR',  'D', '25000'),
    ('S', 'SR',  'D', '30000'),
    ('S', 'SR',  'D', '35000'),
    ('S', 'SR',  'D', '40000'),
    ('S', 'SR',  'D', '45000'),
    ('S', 'SR',  'D', '50000'),
    
    
    
    ('S', 'XR',  'D', '100'),
    ('S', 'XR',  'D', '500'),
    ('S', 'XR',  'D', '1000'),
    ('S', 'XR',  'D', '2000'),
    ('S', 'XR',  'D', '3000'),
    ('S', 'XR',  'D', '4000'),
    ('S', 'XR',  'D', '5000'),
    ('S', 'XR',  'D', '6000'),
    ('S', 'XR',  'D', '7000'),
    ('S', 'XR',  'D', '8000'),
    ('S', 'XR',  'D', '9000'),
    ('S', 'XR',  'D', '10000'),
    ('S', 'XR',  'D', '11000'),
    ('S', 'XR',  'D', '12000'),
    ('S', 'XR',  'D', '13000'),
    ('S', 'XR',  'D', '14000'),
    ('S', 'XR',  'D', '15000'),
    ('S', 'XR',  'D', '16000'),
    ('S', 'XR',  'D', '17000'),
    ('S', 'XR',  'D', '18000'),
    
    ('S', 'LR',  'W', '30'),
    ('S', 'LR',  'W', '50'),
    ('S', 'LR',  'W', '100'),
    ('S', 'LR',  'W', '200'),
    ('S', 'LR',  'W', '300'),
    ('S', 'LR',  'W', '400'),
    ('S', 'LR',  'W', '500'),
    ('S', 'LR',  'W', '600'),
    ('S', 'LR',  'W', '700'),
    ('S', 'LR',  'W', '800'),
    ('S', 'LR',  'W', '900'),
    ('S', 'LR',  'W', '1000')
    ('S', 'LR',  'W', '1100'),
    ('S', 'LR',  'W', '1200'),
    ('S', 'LR',  'W', '1300'),
    
    ('S', 'LR',  'W', '1500'),#too long
    ('S', 'LR',  'W', '2000'),#too long
    ('S', 'LR',  'W', '2500'),#too long
    ('S', 'LR',  'W', '3000'),#too long
    ('S', 'LR',  'W', '3500'),#too long
    ('S', 'LR',  'W', '4000'),#too long
    ('S', 'LR',  'W', '4500'),#too long
    ('S', 'LR',  'W', '5000'),#too long
    ('S', 'LR',  'W', '5500'),#too long
    ('S', 'LR',  'W', '6000'),#too long
    
    ('S', 'RR',  'W', '30'),
    ('S', 'RR',  'W', '50'),
    ('S', 'RR',  'W', '100'),
    ('S', 'RR',  'W', '200'),
    ('S', 'RR',  'W', '300'),
    ('S', 'RR',  'W', '400'),
    ('S', 'RR',  'W', '500'),
    ('S', 'RR',  'W', '600'),
    ('S', 'RR',  'W', '700'),
    ('S', 'RR',  'W', '800'),
    ('S', 'RR',  'W', '900'),
    ('S', 'RR',  'W', '1000'),
    ('S', 'RR',  'W', '1500'),
    ('S', 'RR',  'W', '2000'),
    ('S', 'RR',  'W', '2500'),
    ('S', 'RR',  'W', '3000'),
    ('S', 'RR',  'W', '3500'),
    ('S', 'RR',  'W', '4000'),
    ('S', 'RR',  'W', '4500'),
    ('S', 'RR',  'W', '5000'),
    ('S', 'RR',  'W', '5500'),
    ('S', 'RR',  'W', '6000'),
    ('S', 'RR',  'W', '9000'),
    ('S', 'RR',  'W', '12000'),
    ('S', 'RR',  'W', '15000'),
    ('S', 'RR',  'W', '18000')
    ('S', 'RR',  'W', '21000'),
    ('S', 'RR',  'W', '24000'),
    ('S', 'RR',  'W', '27000'),
    ('S', 'RR',  'W', '30000'),
    ('S', 'RR',  'W', '33000'),
    ('S', 'RR',  'W', '36000'),
    
    ('S', 'XR',  'W', '30'),
    ('S', 'XR',  'W', '250'),
    ('S', 'XR',  'W', '600'),
    ('S', 'XR',  'W', '1000'),
    ('S', 'XR',  'W', '1500'),
    ('S', 'XR',  'W', '2000'),
    ('S', 'XR',  'W', '3000'),
    ('S', 'XR',  'W', '4000'),
    ('S', 'XR',  'W', '6000'),
    ('S', 'XR',  'W', '8000'),
    ('S', 'XR',  'W', '10000'),
    ('S', 'XR',  'W', '12000'),
    ('S', 'XR',  'W', '16000'), 
    
    ('S', 'SR',  'W', '50'),
    ('S', 'SR',  'W', '100'),
    ('S', 'SR',  'W', '200'),
    ('S', 'SR',  'W', '300'),
    ('S', 'SR',  'W', '400'),
    ('S', 'SR',  'W', '500'),
    ('S', 'SR',  'W', '750'),
    ('S', 'SR',  'W', '1000'),
    ('S', 'SR',  'W', '2000'),
    ('S', 'SR',  'W', '3000'),
    ('S', 'SR',  'W', '4000')
    
    #K
    #1,100,500,1000,5000,10000,20000,50000,100000,150000,200000,240000
    ('K', 'LR',  'D', '1'),
    ('K', 'LR',  'D', '100'),
    ('K', 'LR',  'D', '500'),
    ('K', 'LR',  'D', '1000'),
    ('K', 'LR',  'D', '5000'),
    ('K', 'LR',  'D', '10000'),
    ('K', 'LR',  'D', '20000'),
    ('K', 'LR',  'D', '50000'),
    ('K', 'LR',  'D', '100000'),
    ('K', 'LR',  'D', '150000'),
    ('K', 'LR',  'D', '200000'),
    ('K', 'LR',  'D', '240000'),
    
    #1,10,20,30,40,50,75,100,125,150,200,250,300
    ('K', 'RR',  'D', '1'),
    ('K', 'RR',  'D', '10'),
    ('K', 'RR',  'D', '20'),
    ('K', 'RR',  'D', '30'),
    ('K', 'RR',  'D', '40'),
    ('K', 'RR',  'D', '50'),
    ('K', 'RR',  'D', '75'),
    ('K', 'RR',  'D', '100'),
    ('K', 'RR',  'D', '125'),
    ('K', 'RR',  'D', '150'),
    ('K', 'RR',  'D', '200'),
    ('K', 'RR',  'D', '250'),
    ('K', 'RR',  'D', '300'),
    
    #1,10,20,30,40,50,75,100,150,200,250,300,350,380
    ('K', 'SR',  'D', '1'),
    ('K', 'SR',  'D', '10'),
    ('K', 'SR',  'D', '20'),
    ('K', 'SR',  'D', '30'),
    ('K', 'SR',  'D', '40'),
    ('K', 'SR',  'D', '50'),
    ('K', 'SR',  'D', '75'),
    ('K', 'SR',  'D', '100'),
    ('K', 'SR',  'D', '150'),
    ('K', 'SR',  'D', '200'),
    ('K', 'SR',  'D', '250'),
    ('K', 'SR',  'D', '300'),
    ('K', 'SR',  'D', '350'),
    ('K', 'SR',  'D', '380'),
    
    #1,10,25,50,75,100,150,200,250,300,350,400,450
    ('K', 'XR',  'D', '1'),
    ('K', 'XR',  'D', '10'),
    ('K', 'XR',  'D', '25'),
    ('K', 'XR',  'D', '50'),
    ('K', 'XR',  'D', '75'),
    ('K', 'XR',  'D', '100'),
    ('K', 'XR',  'D', '150'),
    ('K', 'XR',  'D', '200'),
    ('K', 'XR',  'D', '250'),
    ('K', 'XR',  'D', '300'),
    ('K', 'XR',  'D', '350'),
    ('K', 'XR',  'D', '400'),
    ('K', 'XR',  'D', '450'),
    
    #1,3,5,10,15,20,30,40,50,60,70
    ('K', 'LR',  'W', '1'),
    ('K', 'LR',  'W', '3'),
    ('K', 'LR',  'W', '5'),
    ('K', 'LR',  'W', '10'),
    ('K', 'LR',  'W', '15'),
    ('K', 'LR',  'W', '20'),
    ('K', 'LR',  'W', '30'),
    ('K', 'LR',  'W', '40'),
    ('K', 'LR',  'W', '50'),
    ('K', 'LR',  'W', '60'),
    ('K', 'LR',  'W', '70'),
    
    #1,2,3,4,5,7,10,15,25,39
    ('K', 'RR',  'W', '1'),
    ('K', 'RR',  'W', '2'),
    ('K', 'RR',  'W', '3'),
    
    ('K', 'RR',  'W', '4'),
    ('K', 'RR',  'W', '5'),
    ('K', 'RR',  'W', '7'),
    ('K', 'RR',  'W', '10'),
    ('K', 'RR',  'W', '15'),
    ('K', 'RR',  'W', '25'),
    ('K', 'RR',  'W', '39'),
    
    ('K', 'SR',  'W', '1'),
    ('K', 'SR',  'W', '10'),
    
    
    #1,3,5,7,10,16,20,26,30,35,40,43
    ('K', 'XR',  'W', '1'),
    ('K', 'XR',  'W', '3'),
    ('K', 'XR',  'W', '5'),
    ('K', 'XR',  'W', '7'),
    ('K', 'XR',  'W', '10'),
    ('K', 'XR',  'W', '15'),
    ('K', 'XR',  'W', '20'),
    ('K', 'XR',  'W', '25'),
    ('K', 'XR',  'W', '30'),
    ('K', 'XR',  'W', '35'),
    ('K', 'XR',  'W', '40'),
    ('K', 'XR',  'W', '43'),
    ('C', 'LR', 'D', '12'),
    ('C', 'LR', 'D', '16'), #ERROR
    ('C', 'LR', 'D', '20'),
    ('C', 'LR', 'D', '26'), #ERROR
    ('C', 'LR', 'D', '30'),
    ('C', 'LR', 'D', '40'),
    ('C', 'LR', 'D', '50'),
    ('C', 'LR', 'D', '60'),
    ('C', 'LR', 'D', '70'),
    ('C', 'LR', 'D', '80'),
    ('C', 'LR', 'D', '90'),
    ('C', 'LR', 'D', '100'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048 
    ('C', 'RR', 'D', '12'),
    ('C', 'RR', 'D', '50'),
    ('C', 'RR', 'D', '100'),
    ('C', 'RR', 'D', '200'),
    ('C', 'RR', 'D', '300'),
    ('C', 'RR', 'D', '400'),
    ('C', 'RR', 'D', '500'),
    ('C', 'RR', 'D', '750'),
    ('C', 'RR', 'D', '1000'),
    ('C', 'RR', 'D', '1250'),
    ('C', 'RR', 'D', '1500'),
    ('C', 'RR', 'D', '1750'),
    ('C', 'RR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'SR', 'D', '12'),
    ('C', 'SR', 'D', '50'),
    ('C', 'SR', 'D', '100'),
    ('C', 'SR', 'D', '200'),
    ('C', 'SR', 'D', '300'),
    ('C', 'SR', 'D', '400')
    
    
        

    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'XR', 'D', '12'),
    ('C', 'XR', 'D', '50'),
    ('C', 'XR', 'D', '100'),
    ('C', 'XR', 'D', '200'),
    ('C', 'XR', 'D', '300'),
    ('C', 'XR', 'D', '400'),
    ('C', 'XR', 'D', '500'),
    ('C', 'XR', 'D', '750'),
    ('C', 'XR', 'D', '1000'),
    ('C', 'XR', 'D', '1250'),
    ('C', 'XR', 'D', '1500'),
    ('C', 'XR', 'D', '1750'),
    ('C', 'XR', 'D', '2048'),
    
    #12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    ('C', 'LR', 'W', '12'),
    ('C', 'LR', 'W', '14'),
    ('C', 'LR', 'W', '16'),
    ('C', 'LR', 'W', '18'),
    ('C', 'LR', 'W', '20'),
    ('C', 'LR', 'W', '22'),
    ('C', 'LR', 'W', '24'),
    ('C', 'LR', 'W', '26'),
    ('C', 'LR', 'W', '28'),
    ('C', 'LR', 'W', '30'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'RR', 'W', '12'),
    ('C', 'RR', 'W', '50'),
    ('C', 'RR', 'W', '100'),
    ('C', 'RR', 'W', '200'),
    ('C', 'RR', 'W', '300'),
    ('C', 'RR', 'W', '400'),
    ('C', 'RR', 'W', '500'),
    ('C', 'RR', 'W', '750'),
    ('C', 'RR', 'W', '1000'),
    ('C', 'RR', 'W', '1250'),
    ('C', 'RR', 'W', '1500'),
    ('C', 'RR', 'W', '1750'),
    ('C', 'RR', 'W', '2048'),

    
    #12, 20,30,40,50,60,70,80,90, 100, 110, 130, 150
    ('C', 'SR', 'W', '12'),
    ('C', 'SR', 'W', '20'),
    ('C', 'SR', 'W', '30'),
    ('C', 'SR', 'W', '40'),
    ('C', 'SR', 'W', '50'),
    ('C', 'SR', 'W', '60'),
    ('C', 'SR', 'W', '70'),
    ('C', 'SR', 'W', '80'),
    ('C', 'SR', 'W', '90'),
    ('C', 'SR', 'W', '100'),
    ('C', 'SR', 'W', '110'), #ERROR
    ('C', 'SR', 'W', '130'),
    ('C', 'SR', 'W', '150'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'XR', 'W', '12'),
    ('C', 'XR', 'W', '50'),
    ('C', 'XR', 'W', '100'),
    ('C', 'XR', 'W', '200'),
    ('C', 'XR', 'W', '300'),
    ('C', 'XR', 'W', '400'),
    ('C', 'XR', 'W', '500'),
    ('C', 'XR', 'W', '750'),
    ('C', 'XR', 'W', '1000'),
    ('C', 'XR', 'W', '1250'),
    ('C', 'XR', 'W', '1500'),
    ('C', 'XR', 'W', '1750'),
    ('C', 'XR', 'W', '2048'),
    
    #12,16,20,26,30,40,50,60,70,80,90,100
    ('G', 'LR', 'D', '12'),
    ('G', 'LR', 'D', '16'), #ERROR
    ('G', 'LR', 'D', '20'),
    ('G', 'LR', 'D', '26'), #ERROR
    ('G', 'LR', 'D', '30'),
    ('G', 'LR', 'D', '40'),
    ('G', 'LR', 'D', '50'),
    ('G', 'LR', 'D', '60'),
    ('G', 'LR', 'D', '70'),
    ('G', 'LR', 'D', '80'),
    ('G', 'LR', 'D', '90'),
    ('G', 'LR', 'D', '100'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'RR', 'D', '12'),
    ('G', 'RR', 'D', '50'),
    ('G', 'RR', 'D', '100'),
    ('G', 'RR', 'D', '200'),
    ('G', 'RR', 'D', '300'),
    ('G', 'RR', 'D', '400'),
    ('G', 'RR', 'D', '500'),
    ('G', 'RR', 'D', '750'),
    ('G', 'RR', 'D', '1000'),
    ('G', 'RR', 'D', '1250'),
    ('G', 'RR', 'D', '1500'),
    ('G', 'RR', 'D', '1750'),
    ('G', 'RR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'SR', 'D', '12'),
    ('G', 'SR', 'D', '50'),
    ('G', 'SR', 'D', '100'),
    ('G', 'SR', 'D', '200'),
    ('G', 'SR', 'D', '300'),
    ('G', 'SR', 'D', '400'),
    ('G', 'SR', 'D', '500'),
    ('G', 'SR', 'D', '750'),
    ('G', 'SR', 'D', '1000'),
    ('G', 'SR', 'D', '1250'),
    ('G', 'SR', 'D', '1500'),
    ('G', 'SR', 'D', '1750'),
    ('G', 'SR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'XR', 'D', '12'),
    ('G', 'XR', 'D', '50'),
    ('G', 'XR', 'D', '100'),
    ('G', 'XR', 'D', '200'),
    ('G', 'XR', 'D', '300'),
    ('G', 'XR', 'D', '400'),
    ('G', 'XR', 'D', '500'),
    ('G', 'XR', 'D', '750'),
    ('G', 'XR', 'D', '1000'),
    ('G', 'XR', 'D', '1250'),
    ('G', 'XR', 'D', '1500'),
    ('G', 'XR', 'D', '1750'),
    ('G', 'XR', 'D', '2048'),
    
    #12, 16, 20, 26, 30, 40, 50, 60, 70, 80
    ('G', 'LR', 'W', '12'),
    ('G', 'LR', 'W', '16'), #ERROR
    ('G', 'LR', 'W', '20'), 
    ('G', 'LR', 'W', '26'), #ERROR
    ('G', 'LR', 'W', '30'),
    ('G', 'LR', 'W', '40'),
    ('G', 'LR', 'W', '50'),
    ('G', 'LR', 'W', '60'),
    ('G', 'LR', 'W', '70'),
    ('G', 'LR', 'W', '80'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'RR', 'W', '12'),
    ('G', 'RR', 'W', '50'),
    ('G', 'RR', 'W', '100'),
    ('G', 'RR', 'W', '200'),
    ('G', 'RR', 'W', '300'),
    ('G', 'RR', 'W', '400'),
    ('G', 'RR', 'W', '500'),
    ('G', 'RR', 'W', '750'),
    ('G', 'RR', 'W', '1000'),
    ('G', 'RR', 'W', '1250'),
    ('G', 'RR', 'W', '1500'),
    ('G', 'RR', 'W', '1750'),
    ('G', 'RR', 'W', '2048'),
    
    #12, 20,30,40,50,60,70,80,90, 100, 110, 130, 150
    ('G', 'SR', 'W', '12'),
    ('G', 'SR', 'W', '20'),
    ('G', 'SR', 'W', '30'),
    ('G', 'SR', 'W', '40'),
    ('G', 'SR', 'W', '50'),
    ('G', 'SR', 'W', '60'),
    ('G', 'SR', 'W', '70'),
    ('G', 'SR', 'W', '80'),
    ('G', 'SR', 'W', '90'),
    ('G', 'SR', 'W', '100'),
    ('G', 'SR', 'W', '110'), #ERROR
    ('G', 'SR', 'W', '130'),
    ('G', 'SR', 'W', '150'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'XR', 'W', '12'),
    ('G', 'XR', 'W', '50'),
    ('G', 'XR', 'W', '100'),
    ('G', 'XR', 'W', '200'),
    ('G', 'XR', 'W', '300'),
    ('G', 'XR', 'W', '400'),
    ('G', 'XR', 'W', '500'),
    ('G', 'XR', 'W', '750'),
    ('G', 'XR', 'W', '1000'),
    ('G', 'XR', 'W', '1250'),
    ('G', 'XR', 'W', '1500'),
    ('G', 'XR', 'W', '1750'),
    ('G', 'XR', 'W', '2048')

    # 12, 14, 16
    ('C', 'LR', 'A', '12'),
    ('C', 'LR', 'A', '14'),
    ('C', 'LR', 'A', '16'),
    
    # 12, 24, 40, 60, 80, 100, 150, 200, 250, 300, 500, 750, 1024
    ('C', 'RR', 'A', '12'),
    ('C', 'RR', 'A', '24'),
    ('C', 'RR', 'A', '40'),
    ('C', 'RR', 'A', '60'),
    ('C', 'RR', 'A', '80'),
    ('C', 'RR', 'A', '100'),
    ('C', 'RR', 'A', '150'),
    ('C', 'RR', 'A', '200'),
    ('C', 'RR', 'A', '250'),
    ('C', 'RR', 'A', '300'),
    ('C', 'RR', 'A', '500'),
    ('C', 'RR', 'A', '750'),
    ('C', 'RR', 'A', '1024'),
    
    # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 200, 300, 500, 750, 1024
    ('C', 'SR', 'A', '10'),
    ('C', 'SR', 'A', '20'),
    ('C', 'SR', 'A', '30'),
    ('C', 'SR', 'A', '40'),
    ('C', 'SR', 'A', '50'),
    ('C', 'SR', 'A', '60'),
    ('C', 'SR', 'A', '70'),
    ('C', 'SR', 'A', '80'),
    ('C', 'SR', 'A', '90'),
    ('C', 'SR', 'A', '100'),
    ('C', 'SR', 'A', '125'),
    ('C', 'SR', 'A', '200'),
    ('C', 'SR', 'A', '300'),
    ('C', 'SR', 'A', '500'),
    ('C', 'SR', 'A', '750'),
    ('C', 'SR', 'A', '1024'),
    
    

    
    # 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140
    ('C', 'XR', 'A', '12'),
    ('C', 'XR', 'A', '20'),
    ('C', 'XR', 'A', '30'),
    ('C', 'XR', 'A', '40'),
    ('C', 'XR', 'A', '50'),
    ('C', 'XR', 'A', '60'),
    ('C', 'XR', 'A', '70'),
    ('C', 'XR', 'A', '80'),
    ('C', 'XR', 'A', '90'),
    ('C', 'XR', 'A', '100'),
    ('C', 'XR', 'A', '110'),
    ('C', 'XR', 'A', '120'),
    ('C', 'XR', 'A', '130'),
    ('C', 'XR', 'A', '140'),
    
    # 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46
    ('G', 'LR', 'A', '12'),
    ('G', 'LR', 'A', '14'),
    ('G', 'LR', 'A', '16'),
    ('G', 'LR', 'A', '18'),  
    ('G', 'LR', 'A', '22'),
    ('G', 'LR', 'A', '24'),
    ('G', 'LR', 'A', '26'),
    ('G', 'LR', 'A', '28'),
    ('G', 'LR', 'A', '30'),
    ('G', 'LR', 'A', '32'),
    ('G', 'LR', 'A', '34'),
    ('G', 'LR', 'A', '36'),
    ('G', 'LR', 'A', '38'),
    ('G', 'LR', 'A', '42'),
    ('G', 'LR', 'A', '44'),
    ('G', 'LR', 'A', '46'),
    
    # 12, 24, 40, 60, 80, 100, 150, 200, 250, 300, 500, 750, 1024
    ('G', 'RR', 'A', '12'),
    ('G', 'RR', 'A', '24'),
    ('G', 'RR', 'A', '40'),
    ('G', 'RR', 'A', '60'),
    ('G', 'RR', 'A', '80'),
    ('G', 'RR', 'A', '100'),
    ('G', 'RR', 'A', '150'),
    ('G', 'RR', 'A', '200'),
    ('G', 'RR', 'A', '250'),
    ('G', 'RR', 'A', '300'),
    ('G', 'RR', 'A', '500'),
    ('G', 'RR', 'A', '750'),
    ('G', 'RR', 'A', '1024'),
    
    # 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175
    ('G', 'SR', 'A', '12'),
    ('G', 'SR', 'A', '20'),
    ('G', 'SR', 'A', '30'),
    ('G', 'SR', 'A', '40'),
    ('G', 'SR', 'A', '60'),
    ('G', 'SR', 'A', '70'),
    ('G', 'SR', 'A', '80'),
    ('G', 'SR', 'A', '90'),
    ('G', 'SR', 'A', '100'),
    ('G', 'SR', 'A', '125'),
    ('G', 'SR', 'A', '150'),
    ('G', 'SR', 'A', '175'),
    ('G', 'SR', 'A', '500'),
    ('G', 'SR', 'A', '1024'),
    
    ('G', 'XR', 'A', '12'),
    ('G', 'XR', 'A', '20'),
    ('G', 'XR', 'A', '30'),
    ('G', 'XR', 'A', '40'),
    ('G', 'XR', 'A', '50'),
    ('G', 'XR', 'A', '60'),
    ('G', 'XR', 'A', '70'),
    ('G', 'XR', 'A', '80'),
    ('G', 'XR', 'A', '90'),
    ('G', 'XR', 'A', '100'),
    ('G', 'XR', 'A', '125'),
    ('G', 'XR', 'A', '150'),
    ('G', 'XR', 'A', '175'),
    ('G', 'XR', 'A', '300'),
    ('G', 'XR', 'A', '600'),
    ('G', 'XR', 'A', '1024'),
    

    
    # 10, 30, 34, 40, 44, 50, 54, 60, 64, 70, 74, 80
    ('K', 'LR', 'A', '10'),
    ('K', 'LR', 'A', '30'),
    ('K', 'LR', 'A', '34'),
    ('K', 'LR', 'A', '40'),
    ('K', 'LR', 'A', '44'),
    ('K', 'LR', 'A', '50'),
    ('K', 'LR', 'A', '54'),
    ('K', 'LR', 'A', '60'),
    ('K', 'LR', 'A', '64'),
    ('K', 'LR', 'A', '70'),
    ('K', 'LR', 'A', '74'),
    ('K', 'LR', 'A', '80'),
    
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    ('K', 'RR', 'A', '1'),
    ('K', 'RR', 'A', '2'),
    ('K', 'RR', 'A', '3'),
    ('K', 'RR', 'A', '4'),
    ('K', 'RR', 'A', '5'),
    ('K', 'RR', 'A', '6'),
    ('K', 'RR', 'A', '7'),
    ('K', 'RR', 'A', '8'),
    ('K', 'RR', 'A', '9'),
    ('K', 'RR', 'A', '10'),
    ('K', 'RR', 'A', '11'),
    ('K', 'RR', 'A', '12'),
    ('K', 'RR', 'A', '13'),
    ('K', 'RR', 'A', '14'),
    ('K', 'RR', 'A', '15'),
    
    # 1, 2, 3, 4, 5
    ('K', 'SR', 'A', '1'),
    ('K', 'SR', 'A', '2'),
    ('K', 'SR', 'A', '3'),
    ('K', 'SR', 'A', '4'),
    
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ('K', 'XR', 'A', '1'),
    ('K', 'XR', 'A', '2'),
    ('K', 'XR', 'A', '3'),
    ('K', 'XR', 'A', '4'),
    ('K', 'XR', 'A', '5'),
    ('K', 'XR', 'A', '6'),
    ('K', 'XR', 'A', '7'),
    ('K', 'XR', 'A', '8'),
    ('K', 'XR', 'A', '9'),
    ('K', 'XR', 'A', '10'),
    ('K', 'XR', 'A', '11'),
    ('K', 'XR', 'A', '12'),
    ('K', 'XR', 'A', '13'),
    ('K', 'XR', 'A', '14'),
    ('K', 'XR', 'A', '15'),
    ('K', 'XR', 'A', '16'),
    ('K', 'XR', 'A', '17'),
    ('K', 'XR', 'A', '18'),
    ('K', 'XR', 'A', '19'),
    
    # 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700
    ('S', 'LR', 'A', '25'),
    ('S', 'LR', 'A', '50'),
    ('S', 'LR', 'A', '75'),
    ('S', 'LR', 'A', '100'),
    ('S', 'LR', 'A', '150'),
    ('S', 'LR', 'A', '200'),
    ('S', 'LR', 'A', '250'),
    ('S', 'LR', 'A', '300'),
    ('S', 'LR', 'A', '400'),
    ('S', 'LR', 'A', '500'),
    ('S', 'LR', 'A', '600'),
    ('S', 'LR', 'A', '700'),
    
    # 10, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000
    ('S', 'RR', 'A', '50'),
    ('S', 'RR', 'A', '100'),
    ('S', 'RR', 'A', '200'),
    ('S', 'RR', 'A', '300'),
    ('S', 'RR', 'A', '400'),
    ('S', 'RR', 'A', '500'),
    ('S', 'RR', 'A', '750'),
    ('S', 'RR', 'A', '1000'),
    ('S', 'RR', 'A', '1250'),
    ('S', 'RR', 'A', '1500'),
    ('S', 'RR', 'A', '1750'),
    ('S', 'RR', 'A', '2000'),
    ('S', 'RR', 'A', '2500'),
    ('S', 'RR', 'A', '3000'),
    ('S', 'RR', 'A', '3500'),
    ('S', 'RR', 'A', '4000'),
    
    # 20, 30, 50, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 2000
    ('S', 'SR', 'A', '20'),
    ('S', 'SR', 'A', '30'),
    ('S', 'SR', 'A', '50'),
    ('S', 'SR', 'A', '80'),
    ('S', 'SR', 'A', '100'),
    ('S', 'SR', 'A', '150'),
    ('S', 'SR', 'A', '200'),
    ('S', 'SR', 'A', '300'),
    ('S', 'SR', 'A', '400'),
    ('S', 'SR', 'A', '500'),
    ('S', 'SR', 'A', '600'),
    ('S', 'SR', 'A', '700'),
    ('S', 'SR', 'A', '800'),
    ('S', 'SR', 'A', '900'),
    ('S', 'SR', 'A', '1000'),
    ('S', 'SR', 'A', '1250'),
    ('S', 'SR', 'A', '1500'),
    ('S', 'SR', 'A', '2000'),
    
    # 20, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000, 6000, 7000
    ('S', 'XR', 'A', '20'),
    ('S', 'XR', 'A', '50'),
    ('S', 'XR', 'A', '100'),
    ('S', 'XR', 'A', '200'),
    ('S', 'XR', 'A', '300'),
    ('S', 'XR', 'A', '400'),
    ('S', 'XR', 'A', '500'),
    ('S', 'XR', 'A', '750'),
    ('S', 'XR', 'A', '1000'),
    ('S', 'XR', 'A', '1500'),
    ('S', 'XR', 'A', '2000'),
    ('S', 'XR', 'A', '2500'),
    ('S', 'XR', 'A', '3000'),
    ('S', 'XR', 'A', '4000'),
    ('S', 'XR', 'A', '5000'),
    ('S', 'XR', 'A', '6000'),
    ('S', 'XR', 'A', '7000'),
    
      #1, 10, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000
    ('K', 'LR', 'M', '1'),
    ('K', 'LR', 'M', '10'),
    ('K', 'LR', 'M', '50'),
    ('K', 'LR', 'M', '100'),
    ('K', 'LR', 'M', '200'),
    ('K', 'LR', 'M', '300'),
    ('K', 'LR', 'M', '400'),
    ('K', 'LR', 'M', '500'),
    ('K', 'LR', 'M', '1000'),
    ('K', 'LR', 'M', '2000'),
    ('K', 'LR', 'M', '3000'),
    ('K', 'LR', 'M', '4000'),
    ('K', 'LR', 'M', '5000'),
    ('K', 'LR', 'M', '6000'),
    ('K', 'LR', 'M', '7000'),
    ('K', 'LR', 'M', '8000'),
    
    #1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250 
    ('K', 'RR', 'M', '1'),
    ('K', 'RR', 'M', '5'),
    ('K', 'RR', 'M', '10'),
    ('K', 'RR', 'M', '20'),
    ('K', 'RR', 'M', '30'),
    ('K', 'RR', 'M', '40'),
    ('K', 'RR', 'M', '50'),
    ('K', 'RR', 'M', '75'),
    ('K', 'RR', 'M', '100'),
    ('K', 'RR', 'M', '150'),
    ('K', 'RR', 'M', '200'),
    ('K', 'RR', 'M', '250'),
    
    #1, 2, 3, 4, 5, 6, 8, 10, 12,
    ('K', 'SR', 'M', '1'),
    ('K', 'SR', 'M', '2'),
    ('K', 'SR', 'M', '3'),
    ('K', 'SR', 'M', '4'),
    ('K', 'SR', 'M', '5'),
    ('K', 'SR', 'M', '6'),
    ('K', 'SR', 'M', '8'),
    ('K', 'SR', 'M', '10'),
    ('K', 'SR', 'M', '12'),
    
    #1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500
    ('K', 'XR', 'M', '1'),
    ('K', 'XR', 'M', '5'),
    ('K', 'XR', 'M', '10'),
    ('K', 'XR', 'M', '25'),
    ('K', 'XR', 'M', '50'),
    ('K', 'XR', 'M', '75'),
    ('K', 'XR', 'M', '100'),
    ('K', 'XR', 'M', '200'),
    ('K', 'XR', 'M', '300'),
    ('K', 'XR', 'M', '400'),
    ('K', 'XR', 'M', '500'),
    ('K', 'XR', 'M', '600'),

    
    #30, 100, 200, 300, 400, 500, 800, 1400, 1500, 3000, 5000, 10000, 20000, 30000, 40000, 50000
    ('S', 'LR', 'M', '30'),
    ('S', 'LR', 'M', '100'),
    ('S', 'LR', 'M', '200'),
    ('S', 'LR', 'M', '300'),
    ('S', 'LR', 'M', '400'),
    ('S', 'LR', 'M', '500'),
    ('S', 'LR', 'M', '800'),
    ('S', 'LR', 'M', '1500'),
    ('S', 'LR', 'M', '3000'),
    ('S', 'LR', 'M', '5000'),
    ('S', 'LR', 'M', '10000'),
    ('S', 'LR', 'M', '20000'),
    ('S', 'LR', 'M', '30000'),
    ('S', 'LR', 'M', '40000'),
    ('S', 'LR', 'M', '50000'),
    
    #30, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000
    ('S', 'RR', 'M', '30'),
    ('S', 'RR', 'M', '100'),
    ('S', 'RR', 'M', '200'),
    ('S', 'RR', 'M', '500'),
    ('S', 'RR', 'M', '1000'),
    ('S', 'RR', 'M', '1500'),
    ('S', 'RR', 'M', '2000'),
    ('S', 'RR', 'M', '3000'),
    ('S', 'RR', 'M', '4000'),
    ('S', 'RR', 'M', '5000'),
    ('S', 'RR', 'M', '6000'),
    ('S', 'RR', 'M', '7000'),
    ('S', 'RR', 'M', '8000'),
    ('S', 'RR', 'M', '9000'),
    ('S', 'RR', 'M', '10000'),
    ('S', 'RR', 'M', '12000'),
    
    #30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 6000
    ('S', 'SR', 'M', '30'),
    ('S', 'SR', 'M', '40'),
    ('S', 'SR', 'M', '50'),
    ('S', 'SR', 'M', '75'),
    ('S', 'SR', 'M', '100'),
    ('S', 'SR', 'M', '150'),
    ('S', 'SR', 'M', '200'),
    ('S', 'SR', 'M', '300'),
    ('S', 'SR', 'M', '500'),
    ('S', 'SR', 'M', '750'),
    ('S', 'SR', 'M', '1000'),
    ('S', 'SR', 'M', '1250'),
    ('S', 'SR', 'M', '1500'),
    ('S', 'SR', 'M', '2000'),
    ('S', 'SR', 'M', '2500'),
    ('S', 'SR', 'M', '3000'),
    ('S', 'SR', 'M', '4000'),
    ('S', 'SR', 'M', '5000'),
    ('S', 'SR', 'M', '6000'),
    
    
    #30, 50, 100, 400, 1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000
    ('S', 'XR', 'M', '150'),
    ('S', 'XR', 'M', '400'),
    ('S', 'XR', 'M', '1000'),
    ('S', 'XR', 'M', '3000'),
    ('S', 'XR', 'M', '5000'),
    ('S', 'XR', 'M', '10000'),
    ('S', 'XR', 'M', '15000'),
    ('S', 'XR', 'M', '20000'),
    ('S', 'XR', 'M', '25000'),
    ('S', 'XR', 'M', '30000'),
    ('S', 'XR', 'M', '35000'),
    ('S', 'XR', 'M', '40000'),
    ('S', 'XR', 'M', '45000'),
    ('S', 'XR', 'M', '50000'),
    

    #16, 50, 100, 200, 300, 400, 1000, 3000, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 32768
    ('C', 'LR', 'M', '16'),
    ('C', 'LR', 'M', '50'),
    ('C', 'LR', 'M', '100'),
    ('C', 'LR', 'M', '200'),
    ('C', 'LR', 'M', '300'),
    ('C', 'LR', 'M', '400'),
    ('C', 'LR', 'M', '1000'),
    ('C', 'LR', 'M', '3000'),
    ('C', 'LR', 'M', '5000'),
    ('C', 'LR', 'M', '7500'),
    ('C', 'LR', 'M', '10000'),
    ('C', 'LR', 'M', '12500'),
    ('C', 'LR', 'M', '15000'),
    ('C', 'LR', 'M', '17500'),
    ('C', 'LR', 'M', '20000'),
    ('C', 'LR', 'M', '25000'),
    ('C', 'LR', 'M', '32768'),
    
    
    #15, 50, 70, 90, 110, 150, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 32768
    ('C', 'RR', 'M', '16'),
    ('C', 'RR', 'M', '50'),
    ('C', 'RR', 'M', '70'),
    ('C', 'RR', 'M', '90'),
    ('C', 'RR', 'M', '110'),
    ('C', 'RR', 'M', '150'),
    ('C', 'RR', 'M', '500'),
    ('C', 'RR', 'M', '1000'),
    ('C', 'RR', 'M', '2000'),
    ('C', 'RR', 'M', '3000'),
    ('C', 'RR', 'M', '5000'),
    ('C', 'RR', 'M', '7500'),
    ('C', 'RR', 'M', '10000'),
    ('C', 'RR', 'M', '15000'),
    ('C', 'RR', 'M', '20000'),
    ('C', 'RR', 'M', '25000'),
    ('C', 'RR', 'M', '32768'),
    
    #16, 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 32768
    ('C', 'SR', 'M', '16'),
    ('C', 'SR', 'M', '50'),
    ('C', 'SR', 'M', '100'),
    ('C', 'SR', 'M', '250'),
    ('C', 'SR', 'M', '500'),
    ('C', 'SR', 'M', '1000'),
    ('C', 'SR', 'M', '2500'),
    ('C', 'SR', 'M', '5000'),
    ('C', 'SR', 'M', '7500'),
    ('C', 'SR', 'M', '10000'),
    ('C', 'SR', 'M', '12500'),
    ('C', 'SR', 'M', '15000'),
    ('C', 'SR', 'M', '20000'),
    ('C', 'SR', 'M', '25000'),
    ('C', 'SR', 'M', '32768'),
    
    #16, 50, 150, 300, 500, 700, 1000, 1250, 1500, 2000, 3000, 4000, 5000, 7500, 10000
    ('C', 'XR', 'M', '16'),
    ('C', 'XR', 'M', '50'),
    ('C', 'XR', 'M', '150'),
    ('C', 'XR', 'M', '300'),
    ('C', 'XR', 'M', '500'),
    ('C', 'XR', 'M', '700'),
    ('C', 'XR', 'M', '1000'),
    ('C', 'XR', 'M', '1250'),
    ('C', 'XR', 'M', '1500'),
    ('C', 'XR', 'M', '2000'),
    ('C', 'XR', 'M', '3000'),
    ('C', 'XR', 'M', '4000'),
    ('C', 'XR', 'M', '5000'),
    ('C', 'XR', 'M', '7500'),
    ('C', 'XR', 'M', '10000'),
    
 
    
    #16, 50, 100, 250, 500, 1000, 2000, 5000, 7500, 10000, 15000, 20000, 25000, 32768
    ('G', 'LR', 'M', '16'), 
    ('G', 'LR', 'M', '50'),
    ('G', 'LR', 'M', '100'),
    ('G', 'LR', 'M', '250'),
    ('G', 'LR', 'M', '500'),
    ('G', 'LR', 'M', '1000'),
    ('G', 'LR', 'M', '2000'),
    ('G', 'LR', 'M', '5000'),
    ('G', 'LR', 'M', '7500'),
    ('G', 'LR', 'M', '10000'),
    ('G', 'LR', 'M', '15000'),
    ('G', 'LR', 'M', '20000'),
    ('G', 'LR', 'M', '25000'),
    ('G', 'LR', 'M', '32768'),
 
    
    # 16, 50, 60, 80, 100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 32768
    ('G', 'RR', 'M', '16'),
    ('G', 'RR', 'M', '50'),
    ('G', 'RR', 'M', '60'),
    ('G', 'RR', 'M', '80'),
    ('G', 'RR', 'M', '100'),
    ('G', 'RR', 'M', '500'),
    ('G', 'RR', 'M', '1000'),
    ('G', 'RR', 'M', '2000'),
    ('G', 'RR', 'M', '3000'),
    ('G', 'RR', 'M', '5000'),
    ('G', 'RR', 'M', '7500'),
    ('G', 'RR', 'M', '10000'),
    ('G', 'RR', 'M', '15000'),
    ('G', 'RR', 'M', '20000'),
    ('G', 'RR', 'M', '25000'),
    ('G', 'RR', 'M', '32768'),

    
    #16, 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 32768
    ('G', 'SR', 'M', '16'),
    ('G', 'SR', 'M', '50'),
    ('G', 'SR', 'M', '100'),
    ('G', 'SR', 'M', '250'),
    ('G', 'SR', 'M', '500'),
    ('G', 'SR', 'M', '1000'),
    ('G', 'SR', 'M', '2500'),
    ('G', 'SR', 'M', '5000'),
    ('G', 'SR', 'M', '7500'),
    ('G', 'SR', 'M', '10000'),
    ('G', 'SR', 'M', '12500'),
    ('G', 'SR', 'M', '15000'),
    ('G', 'SR', 'M', '20000'),
    ('G', 'SR', 'M', '25000'),
    ('G', 'SR', 'M', '32768'),
    
    # 16, 50, 70, 90, 150, 300, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000
    ('G', 'XR', 'M', '16'),
    ('G', 'XR', 'M', '50'),
    ('G', 'XR', 'M', '70'),
    ('G', 'XR', 'M', '90'),
    ('G', 'XR', 'M', '150'),
    ('G', 'XR', 'M', '300'),
    ('G', 'XR', 'M', '500'),
    ('G', 'XR', 'M', '1000'),
    ('G', 'XR', 'M', '2500'),
    ('G', 'XR', 'M', '5000'),
    ('G', 'XR', 'M', '7500'),
    ('G', 'XR', 'M', '10000'),
    ('G', 'XR', 'M', '12500'),
    ('G', 'XR', 'M', '15000'),


    
    '''
    
    run_queue = [
    ('K', 'LR', 'M', '8000'),
    
        
    ('C', 'LR', 'M', '1000'),
    ('C', 'LR', 'M', '3000'),
    ('C', 'LR', 'M', '5000'),
    ('C', 'LR', 'M', '7500'),
    ('C', 'LR', 'M', '10000'),
    ('C', 'LR', 'M', '12500'),
    ('C', 'LR', 'M', '15000'),
    ('C', 'LR', 'M', '17500'),
    ('C', 'LR', 'M', '20000'),
    ('C', 'LR', 'M', '25000'),
    ('C', 'LR', 'M', '32768'),
    
    
    
    ('G', 'LR', 'M', '16'), 
    ('G', 'LR', 'M', '50'),
    ('G', 'LR', 'M', '100'),
    ('G', 'LR', 'M', '250'),
    ('G', 'LR', 'M', '500'),
    ('G', 'LR', 'M', '1000'),
    ('G', 'LR', 'M', '2000'),
    ('G', 'LR', 'M', '5000'),
    ('G', 'LR', 'M', '7500'),
    ('G', 'LR', 'M', '10000'),
    ('G', 'LR', 'M', '15000'),
    ('G', 'LR', 'M', '20000'),
    ('G', 'LR', 'M', '25000'),
    ('G', 'LR', 'M', '32768'),
    
    
    ('C', 'RR', 'M', '500'),
    ('C', 'RR', 'M', '1000'),
    ('C', 'RR', 'M', '2000'),
    ('C', 'RR', 'M', '3000'),
    ('C', 'RR', 'M', '5000'),
    ('C', 'RR', 'M', '7500'),
    ('C', 'RR', 'M', '10000'),
    ('C', 'RR', 'M', '15000'),
    ('C', 'RR', 'M', '20000'),
    ('C', 'RR', 'M', '25000'),
    ('C', 'RR', 'M', '32768'),
    
    ('S', 'RR', 'M', '30'),
    ('S', 'RR', 'M', '100'),
    ('S', 'RR', 'M', '200'),
    ('S', 'RR', 'M', '500'),
    ('S', 'RR', 'M', '1000'),
    ('S', 'RR', 'M', '1500'),
    ('S', 'RR', 'M', '2000'),
    ('S', 'RR', 'M', '3000'),
    ('S', 'RR', 'M', '4000'),
    ('S', 'RR', 'M', '5000'),
    ('S', 'RR', 'M', '6000'),
    ('S', 'RR', 'M', '7000'),
    ('S', 'RR', 'M', '8000'),
    ('S', 'RR', 'M', '9000'),
    ('S', 'RR', 'M', '10000'),
    ('S', 'RR', 'M', '12000'),
    
    ('G', 'RR', 'M', '16'),
    ('G', 'RR', 'M', '50'),
    ('G', 'RR', 'M', '60'),
    ('G', 'RR', 'M', '80'),
    ('G', 'RR', 'M', '100'),
    ('G', 'RR', 'M', '500'),
    ('G', 'RR', 'M', '1000'),
    ('G', 'RR', 'M', '2000'),
    ('G', 'RR', 'M', '3000'),
    ('G', 'RR', 'M', '5000'),
    ('G', 'RR', 'M', '7500'),
    ('G', 'RR', 'M', '10000'),
    ('G', 'RR', 'M', '15000'),
    ('G', 'RR', 'M', '20000'),
    ('G', 'RR', 'M', '25000'),
    ('G', 'RR', 'M', '32768'),
    
    ('S', 'SR', 'M', '4000'),
    ('S', 'SR', 'M', '5000'),
    ('S', 'SR', 'M', '6000'),
    
    ('C', 'SR', 'M', '16'),
    ('C', 'SR', 'M', '50'),
    ('C', 'SR', 'M', '100'),
    ('C', 'SR', 'M', '250'),
    ('C', 'SR', 'M', '500'),
    ('C', 'SR', 'M', '1000'),
    ('C', 'SR', 'M', '2500'),
    ('C', 'SR', 'M', '5000'),
    ('C', 'SR', 'M', '7500'),
    ('C', 'SR', 'M', '10000'),
    ('C', 'SR', 'M', '12500'),
    ('C', 'SR', 'M', '15000'),
    ('C', 'SR', 'M', '20000'),
    ('C', 'SR', 'M', '25000'),
    ('C', 'SR', 'M', '32768'),
    
    ('G', 'SR', 'M', '16'),
    ('G', 'SR', 'M', '50'),
    ('G', 'SR', 'M', '100'),
    ('G', 'SR', 'M', '250'),
    ('G', 'SR', 'M', '500'),
    ('G', 'SR', 'M', '1000'),
    ('G', 'SR', 'M', '2500'),
    ('G', 'SR', 'M', '5000'),
    ('G', 'SR', 'M', '7500'),
    ('G', 'SR', 'M', '10000'),
    ('G', 'SR', 'M', '12500'),
    ('G', 'SR', 'M', '15000'),
    ('G', 'SR', 'M', '20000'),
    ('G', 'SR', 'M', '25000'),
    ('G', 'SR', 'M', '32768'),

    
    ('C', 'XR', 'M', '3000'),
    ('C', 'XR', 'M', '4000'),
    ('C', 'XR', 'M', '5000'),
    ('C', 'XR', 'M', '7500'),
    ('C', 'XR', 'M', '10000'),
    
    ('G', 'XR', 'M', '150'),
    ('G', 'XR', 'M', '300'),
    ('G', 'XR', 'M', '500'),
    ('G', 'XR', 'M', '1000'),
    ('G', 'XR', 'M', '2500'),
    ('G', 'XR', 'M', '5000'),
    ('G', 'XR', 'M', '7500'),
    ('G', 'XR', 'M', '10000'),
    ('G', 'XR', 'M', '12500'),
    ('G', 'XR', 'M', '15000'),

    ('S', 'LR', 'A', '10'),
    ('S', 'LR', 'A', '25'),
    ('S', 'LR', 'A', '50'),
    ('S', 'LR', 'A', '75'),
    ('S', 'LR', 'A', '100'),
    ('S', 'LR', 'A', '150'),
    ('S', 'LR', 'A', '200'),
    ('S', 'LR', 'A', '250'),
    ('S', 'LR', 'A', '300'),
    ('S', 'LR', 'A', '400'),
    ('S', 'LR', 'A', '500'),
    ('S', 'LR', 'A', '600'),
    ('S', 'LR', 'A', '700'),
    
    ('C', 'RR', 'A', '12'),
    ('C', 'RR', 'A', '24'),
    ('C', 'RR', 'A', '40'),
    ('C', 'RR', 'A', '60'),
    ('C', 'RR', 'A', '80'),
    ('C', 'RR', 'A', '100'),
    ('C', 'RR', 'A', '150'),
    ('C', 'RR', 'A', '200'),
    ('C', 'RR', 'A', '250'),
    ('C', 'RR', 'A', '300'),
    ('C', 'RR', 'A', '500'),
    ('C', 'RR', 'A', '750'),
    ('C', 'RR', 'A', '1024'),
    
    ('G', 'RR', 'A', '12'),
    ('G', 'RR', 'A', '24'),
    ('G', 'RR', 'A', '40'),
    ('G', 'RR', 'A', '60'),
    ('G', 'RR', 'A', '80'),
    ('G', 'RR', 'A', '100'),
    ('G', 'RR', 'A', '150'),
    ('G', 'RR', 'A', '200'),
    ('G', 'RR', 'A', '250'),
    ('G', 'RR', 'A', '300'),
    ('G', 'RR', 'A', '500'),
    ('G', 'RR', 'A', '750'),
    ('G', 'RR', 'A', '1024'),
    
    ('G', 'SR', 'A', '500'),
    ('G', 'SR', 'A', '1024'),
    
    ('C', 'SR', 'A', '200'),
    ('C', 'SR', 'A', '300'),
    ('C', 'SR', 'A', '500'),
    ('C', 'SR', 'A', '750'),
    ('C', 'SR', 'A', '1024'),
    
    ('G', 'XR', 'A', '300'),
    ('G', 'XR', 'A', '600'),
    ('G', 'XR', 'A', '1024'),
  
    ]
    
    
    for approximation, model, data, n_samples in run_queue:
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
    
  
def run_evaluations():
    
    running_counter = 0
    src = str(src_dir)

    # (approximation, model, n_samples) – sortiert nach geschätzter Runtime (schnelle zuerst)
    '''
    ('S', 'LR',  'D', '100'),
    ('S', 'LR',  'D', '200'),
    ('S', 'LR',  'D', '300'),
    ('S', 'LR',  'D', '400'),
    ('S', 'LR',  'D', '500'),
    ('S', 'LR',  'D', '600'),
    ('S', 'LR',  'D', '700'),
    ('S', 'LR',  'D', '800'),
    ('S', 'LR',  'D', '900'),
    ('S', 'LR',  'D', '1000'),
    ('S', 'LR',  'D', '1500'),
    ('S', 'LR',  'D', '2000'),
    ('S', 'LR',  'D', '2500'),

    ('S', 'RR',  'D', '100'),
    ('S', 'RR',  'D', '700'),
    ('S', 'RR',  'D', '1000'),
    ('S', 'RR',  'D', '1150'),
    ('S', 'RR',  'D', '1300'),
    ('S', 'RR',  'D', '1500'),
    ('S', 'RR',  'D', '3000'),
    ('S', 'RR',  'D', '6000'),
    ('S', 'RR',  'D', '7000'),
    ('S', 'RR',  'D', '8000'),
    ('S', 'RR',  'D', '9000'),
    ('S', 'RR',  'D', '10000'),
    ('S', 'RR',  'D', '12000'),
    ('S', 'RR',  'D', '14000'),
    ('S', 'RR',  'D', '16000'),
    
    ('S', 'SR',  'D', '50'),
    ('S', 'SR',  'D', '500'),
    ('S', 'SR',  'D', '1000'),
    ('S', 'SR',  'D', '3000'),
    ('S', 'SR',  'D', '5000'),
    ('S', 'SR',  'D', '10000'),
    ('S', 'SR',  'D', '15000'),
    ('S', 'SR',  'D', '20000'),
    ('S', 'SR',  'D', '25000'),
    ('S', 'SR',  'D', '30000'),
    ('S', 'SR',  'D', '35000'),
    ('S', 'SR',  'D', '40000'),
    ('S', 'SR',  'D', '45000'),
    ('S', 'SR',  'D', '50000'),
    
    ('S', 'XR',  'D', '100'),
    ('S', 'XR',  'D', '500'),
    ('S', 'XR',  'D', '1000'),
    ('S', 'XR',  'D', '2000'),
    ('S', 'XR',  'D', '3000'),
    ('S', 'XR',  'D', '4000'),
    ('S', 'XR',  'D', '5000'),
    ('S', 'XR',  'D', '6000'),
    ('S', 'XR',  'D', '7000'),
    ('S', 'XR',  'D', '8000'),
    ('S', 'XR',  'D', '9000'),
    ('S', 'XR',  'D', '10000'),
    ('S', 'XR',  'D', '11000'),
    ('S', 'XR',  'D', '12000'),
    ('S', 'XR',  'D', '13000'),
    ('S', 'XR',  'D', '14000'),
    ('S', 'XR',  'D', '15000'),
    ('S', 'XR',  'D', '16000'),
    ('S', 'XR',  'D', '17000'),
    ('S', 'XR',  'D', '18000'),
    
    ('S', 'LR',  'W', '30'),
    ('S', 'LR',  'W', '50'),
    ('S', 'LR',  'W', '100'),
    ('S', 'LR',  'W', '200'),
    ('S', 'LR',  'W', '300'),
    ('S', 'LR',  'W', '400'),
    ('S', 'LR',  'W', '500'),
    ('S', 'LR',  'W', '600'),
    ('S', 'LR',  'W', '700'),
    ('S', 'LR',  'W', '800'),
    ('S', 'LR',  'W', '900'),
    ('S', 'LR',  'W', '1000'),
    ('S', 'LR',  'W', '1100'),
    ('S', 'LR',  'W', '1200'),
    ('S', 'LR',  'W', '1300'),
    
    ('S', 'RR',  'W', '30'),
    ('S', 'RR',  'W', '300'),
    ('S', 'RR',  'W', '1000'),
    ('S', 'RR',  'W', '1500'),
    ('S', 'RR',  'W', '2500'),
    ('S', 'RR',  'W', '3000'),
    ('S', 'RR',  'W', '3500'),
    ('S', 'RR',  'W', '9000'),
    ('S', 'RR',  'W', '12000'),
    ('S', 'RR',  'W', '15000'),
    ('S', 'RR',  'W', '18000'),
    ('S', 'RR',  'W', '21000'),
    ('S', 'RR',  'W', '24000'),
    ('S', 'RR',  'W', '27000'),
    ('S', 'RR',  'W', '30000'),
    ('S', 'RR',  'W', '33000'),
    ('S', 'RR',  'W', '36000'),
    
    ('S', 'SR',  'W', '50'),
    ('S', 'SR',  'W', '100'),
    ('S', 'SR',  'W', '200'),
    ('S', 'SR',  'W', '300'),
    ('S', 'SR',  'W', '400'),
    ('S', 'SR',  'W', '500'),
    ('S', 'SR',  'W', '750'),
    ('S', 'SR',  'W', '1000'),
    ('S', 'SR',  'W', '2000'),
    ('S', 'SR',  'W', '3000'),
    ('S', 'SR',  'W', '4000'),
    
    ('S', 'XR',  'W', '30'),
    ('S', 'XR',  'W', '250'),
    ('S', 'XR',  'W', '600'),
    ('S', 'XR',  'W', '1000'),
    ('S', 'XR',  'W', '1500'),
    ('S', 'XR',  'W', '2000'),
    ('S', 'XR',  'W', '3000'),
    ('S', 'XR',  'W', '4000'),
    ('S', 'XR',  'W', '6000'),
    ('S', 'XR',  'W', '8000'),
    ('S', 'XR',  'W', '10000'),
    ('S', 'XR',  'W', '12000'),
    ('S', 'XR',  'W', '16000'),
    
    ('K', 'LR',  'D', '1'),
    ('K', 'LR',  'D', '100'),
    ('K', 'LR',  'D', '500'),
    ('K', 'LR',  'D', '1000'),
    ('K', 'LR',  'D', '5000'),
    ('K', 'LR',  'D', '10000'),
    ('K', 'LR',  'D', '20000'),
    ('K', 'LR',  'D', '50000'),
    ('K', 'LR',  'D', '100000'),
    ('K', 'LR',  'D', '150000'),
    ('K', 'LR',  'D', '200000'),
    ('K', 'LR',  'D', '240000'),
    
    #1,10,20,30,40,50,75,100,125,150,200,250,300
    ('K', 'RR',  'D', '1'),
    ('K', 'RR',  'D', '10'),
    ('K', 'RR',  'D', '20'),
    ('K', 'RR',  'D', '30'),
    ('K', 'RR',  'D', '40'),
    ('K', 'RR',  'D', '50'),
    ('K', 'RR',  'D', '75'),
    ('K', 'RR',  'D', '100'),
    ('K', 'RR',  'D', '125'),
    ('K', 'RR',  'D', '150'),
    ('K', 'RR',  'D', '200'),
    ('K', 'RR',  'D', '250'),
    ('K', 'RR',  'D', '300'),
    
    #1,10,20,30,40,50,75,100,150,200,250,300,350,380
    ('K', 'SR',  'D', '1'),
    ('K', 'SR',  'D', '10'),
    ('K', 'SR',  'D', '20'),
    ('K', 'SR',  'D', '30'),
    ('K', 'SR',  'D', '40'),
    ('K', 'SR',  'D', '50'),
    ('K', 'SR',  'D', '75'),
    ('K', 'SR',  'D', '100'),
    ('K', 'SR',  'D', '150'),
    ('K', 'SR',  'D', '200'),
    ('K', 'SR',  'D', '250'),
    ('K', 'SR',  'D', '300'),
    ('K', 'SR',  'D', '350'),
    ('K', 'SR',  'D', '380'),
    
    #1,10,25,50,75,100,150,200,250,300,350,400,450
    ('K', 'XR',  'D', '1'),
    ('K', 'XR',  'D', '10'),
    ('K', 'XR',  'D', '25'),
    ('K', 'XR',  'D', '50'),
    ('K', 'XR',  'D', '75'),
    ('K', 'XR',  'D', '100'),
    ('K', 'XR',  'D', '150'),
    ('K', 'XR',  'D', '200'),
    ('K', 'XR',  'D', '250'),
    ('K', 'XR',  'D', '300'),
    ('K', 'XR',  'D', '350'),
    ('K', 'XR',  'D', '400'),
    ('K', 'XR',  'D', '450'),
    
    #1,3,5,10,15,20,30,40,50,60,70
    ('K', 'LR',  'W', '1'),
    ('K', 'LR',  'W', '3'),
    ('K', 'LR',  'W', '5'),
    ('K', 'LR',  'W', '10'),
    ('K', 'LR',  'W', '15'),
    ('K', 'LR',  'W', '20'),
    ('K', 'LR',  'W', '30'),
    ('K', 'LR',  'W', '40'),
    ('K', 'LR',  'W', '50'),
    ('K', 'LR',  'W', '60'),
    ('K', 'LR',  'W', '70'),
    
    #1,2,3,4,5,7,10,15,25,39
    ('K', 'RR',  'W', '1'),
    ('K', 'RR',  'W', '2'),
    ('K', 'RR',  'W', '3'),
    ('K', 'RR',  'W', '4'),
    ('K', 'RR',  'W', '5'),
    ('K', 'RR',  'W', '7'),
    ('K', 'RR',  'W', '10'),
    ('K', 'RR',  'W', '15'),
    ('K', 'RR',  'W', '25'),
    ('K', 'RR',  'W', '39'),
    
    ('K', 'SR',  'W', '1'),
    ('K', 'SR',  'W', '10'),
    
    #1,3,5,7,10,15,20,25,30,35,40,43
    ('K', 'XR',  'W', '1'),
    ('K', 'XR',  'W', '3'),
    ('K', 'XR',  'W', '5'),
    ('K', 'XR',  'W', '7'),
    ('K', 'XR',  'W', '10'),
    ('K', 'XR',  'W', '15'),
    ('K', 'XR',  'W', '20'),
    ('K', 'XR',  'W', '25'),
    ('K', 'XR',  'W', '30'),
    ('K', 'XR',  'W', '35'),
    ('K', 'XR',  'W', '40'),
    ('K', 'XR',  'W', '43'),
    ('S', 'LR',  'D', '100'),
    ('S', 'LR',  'D', '200'),
    ('S', 'LR',  'D', '300'),
    ('S', 'LR',  'D', '400'),
    ('S', 'LR',  'D', '500'),
    ('S', 'LR',  'D', '600'),
    ('S', 'LR',  'D', '700'),
    ('S', 'LR',  'D', '800'),
    ('S', 'LR',  'D', '900'),
    ('S', 'LR',  'D', '1000'),
    ('S', 'LR',  'D', '1500'),
    ('S', 'LR',  'D', '2000'),
    ('S', 'LR',  'D', '2500'),

    ('S', 'RR',  'D', '100'),
    ('S', 'RR',  'D', '700'),
    ('S', 'RR',  'D', '1000'),
    ('S', 'RR',  'D', '1150'),
    ('S', 'RR',  'D', '1300'),
    ('S', 'RR',  'D', '1500'),
    ('S', 'RR',  'D', '3000'),
    ('S', 'RR',  'D', '6000'),
    ('S', 'RR',  'D', '7000'),
    ('S', 'RR',  'D', '8000'),
    ('S', 'RR',  'D', '9000'),
    ('S', 'RR',  'D', '10000'),
    ('S', 'RR',  'D', '12000'),
    ('S', 'RR',  'D', '14000'),
    ('S', 'RR',  'D', '16000'),
    
    ('S', 'SR',  'D', '50'),
    ('S', 'SR',  'D', '500'),
    ('S', 'SR',  'D', '1000'),
    ('S', 'SR',  'D', '3000'),
    ('S', 'SR',  'D', '5000'),
    ('S', 'SR',  'D', '10000'),
    ('S', 'SR',  'D', '15000'),
    ('S', 'SR',  'D', '20000'),
    ('S', 'SR',  'D', '25000'),
    ('S', 'SR',  'D', '30000'),
    ('S', 'SR',  'D', '35000'),
    ('S', 'SR',  'D', '40000'),
    ('S', 'SR',  'D', '45000'),
    ('S', 'SR',  'D', '50000'),
    
    ('S', 'XR',  'D', '100'),
    ('S', 'XR',  'D', '500'),
    ('S', 'XR',  'D', '1000'),
    ('S', 'XR',  'D', '2000'),
    ('S', 'XR',  'D', '3000'),
    ('S', 'XR',  'D', '4000'),
    ('S', 'XR',  'D', '5000'),
    ('S', 'XR',  'D', '6000'),
    ('S', 'XR',  'D', '7000'),
    ('S', 'XR',  'D', '8000'),
    ('S', 'XR',  'D', '9000'),
    ('S', 'XR',  'D', '10000'),
    ('S', 'XR',  'D', '11000'),
    ('S', 'XR',  'D', '12000'),
    ('S', 'XR',  'D', '13000'),
    ('S', 'XR',  'D', '14000'),
    ('S', 'XR',  'D', '15000'),
    ('S', 'XR',  'D', '16000'),
    ('S', 'XR',  'D', '17000'),
    ('S', 'XR',  'D', '18000'),
    
    ('S', 'LR',  'W', '30'),
    ('S', 'LR',  'W', '50'),
    ('S', 'LR',  'W', '100'),
    ('S', 'LR',  'W', '200'),
    ('S', 'LR',  'W', '300'),
    ('S', 'LR',  'W', '400'),
    ('S', 'LR',  'W', '500'),
    ('S', 'LR',  'W', '600'),
    ('S', 'LR',  'W', '700'),
    ('S', 'LR',  'W', '800'),
    ('S', 'LR',  'W', '900'),
    ('S', 'LR',  'W', '1000'),
    ('S', 'LR',  'W', '1100'),
    ('S', 'LR',  'W', '1200'),
    ('S', 'LR',  'W', '1300'),
    
    ('S', 'RR',  'W', '30'),
    ('S', 'RR',  'W', '300'),
    ('S', 'RR',  'W', '1000'),
    ('S', 'RR',  'W', '1500'),
    ('S', 'RR',  'W', '2500'),
    ('S', 'RR',  'W', '3000'),
    ('S', 'RR',  'W', '3500'),
    ('S', 'RR',  'W', '9000'),
    ('S', 'RR',  'W', '12000'),
    ('S', 'RR',  'W', '15000'),
    ('S', 'RR',  'W', '18000'),
    ('S', 'RR',  'W', '21000'),
    ('S', 'RR',  'W', '24000'),
    ('S', 'RR',  'W', '27000'),
    ('S', 'RR',  'W', '30000'),
    ('S', 'RR',  'W', '33000'),
    ('S', 'RR',  'W', '36000'),
    
    ('S', 'SR',  'W', '50'),
    ('S', 'SR',  'W', '100'),
    ('S', 'SR',  'W', '200'),
    ('S', 'SR',  'W', '300'),
    ('S', 'SR',  'W', '400'),
    ('S', 'SR',  'W', '500'),
    ('S', 'SR',  'W', '750'),
    ('S', 'SR',  'W', '1000'),
    ('S', 'SR',  'W', '2000'),
    ('S', 'SR',  'W', '3000'),
    ('S', 'SR',  'W', '4000'),
    
    ('S', 'XR',  'W', '30'),
    ('S', 'XR',  'W', '250'),
    ('S', 'XR',  'W', '600'),
    ('S', 'XR',  'W', '1000'),
    ('S', 'XR',  'W', '1500'),
    ('S', 'XR',  'W', '2000'),
    ('S', 'XR',  'W', '3000'),
    ('S', 'XR',  'W', '4000'),
    ('S', 'XR',  'W', '6000'),
    ('S', 'XR',  'W', '8000'),
    ('S', 'XR',  'W', '10000'),
    ('S', 'XR',  'W', '12000'),
    ('S', 'XR',  'W', '16000'),
    
    ('K', 'LR',  'D', '1'),
    ('K', 'LR',  'D', '100'),
    ('K', 'LR',  'D', '500'),
    ('K', 'LR',  'D', '1000'),
    ('K', 'LR',  'D', '5000'),
    ('K', 'LR',  'D', '10000'),
    ('K', 'LR',  'D', '20000'),
    ('K', 'LR',  'D', '50000'),
    ('K', 'LR',  'D', '100000'),
    ('K', 'LR',  'D', '150000'),
    ('K', 'LR',  'D', '200000'),
    ('K', 'LR',  'D', '240000'),
    
    #1,10,20,30,40,50,75,100,125,150,200,250,300
    ('K', 'RR',  'D', '1'),
    ('K', 'RR',  'D', '10'),
    ('K', 'RR',  'D', '20'),
    ('K', 'RR',  'D', '30'),
    ('K', 'RR',  'D', '40'),
    ('K', 'RR',  'D', '50'),
    ('K', 'RR',  'D', '75'),
    ('K', 'RR',  'D', '100'),
    ('K', 'RR',  'D', '125'),
    ('K', 'RR',  'D', '150'),
    ('K', 'RR',  'D', '200'),
    ('K', 'RR',  'D', '250'),
    ('K', 'RR',  'D', '300'),
    
    #1,10,20,30,40,50,75,100,150,200,250,300,350,380
    ('K', 'SR',  'D', '1'),
    ('K', 'SR',  'D', '10'),
    ('K', 'SR',  'D', '20'),
    ('K', 'SR',  'D', '30'),
    ('K', 'SR',  'D', '40'),
    ('K', 'SR',  'D', '50'),
    ('K', 'SR',  'D', '75'),
    ('K', 'SR',  'D', '100'),
    ('K', 'SR',  'D', '150'),
    ('K', 'SR',  'D', '200'),
    ('K', 'SR',  'D', '250'),
    ('K', 'SR',  'D', '300'),
    ('K', 'SR',  'D', '350'),
    ('K', 'SR',  'D', '380'),
    
    #1,10,25,50,75,100,150,200,250,300,350,400,450
    ('K', 'XR',  'D', '1'),
    ('K', 'XR',  'D', '10'),
    ('K', 'XR',  'D', '25'),
    ('K', 'XR',  'D', '50'),
    ('K', 'XR',  'D', '75'),
    ('K', 'XR',  'D', '100'),
    ('K', 'XR',  'D', '150'),
    ('K', 'XR',  'D', '200'),
    ('K', 'XR',  'D', '250'),
    ('K', 'XR',  'D', '300'),
    ('K', 'XR',  'D', '350'),
    ('K', 'XR',  'D', '400'),
    ('K', 'XR',  'D', '450'),
    
    #1,3,5,10,15,20,30,40,50,60,70
    ('K', 'LR',  'W', '1'),
    ('K', 'LR',  'W', '3'),
    ('K', 'LR',  'W', '5'),
    ('K', 'LR',  'W', '10'),
    ('K', 'LR',  'W', '15'),
    ('K', 'LR',  'W', '20'),
    ('K', 'LR',  'W', '30'),
    ('K', 'LR',  'W', '40'),
    ('K', 'LR',  'W', '50'),
    ('K', 'LR',  'W', '60'),
    ('K', 'LR',  'W', '70'),
    
    #1,2,3,4,5,7,10,15,25,39
    ('K', 'RR',  'W', '1'),
    ('K', 'RR',  'W', '2'),
    ('K', 'RR',  'W', '3'),
    ('K', 'RR',  'W', '4'),
    ('K', 'RR',  'W', '5'),
    ('K', 'RR',  'W', '7'),
    ('K', 'RR',  'W', '10'),
    ('K', 'RR',  'W', '15'),
    ('K', 'RR',  'W', '25'),
    ('K', 'RR',  'W', '39'),
    
    ('K', 'SR',  'W', '1'),
    ('K', 'SR',  'W', '10'),
    
    #1,3,5,7,10,15,20,25,30,35,40,43
    ('K', 'XR',  'W', '1'),
    ('K', 'XR',  'W', '3'),
    ('K', 'XR',  'W', '5'),
    ('K', 'XR',  'W', '7'),
    ('K', 'XR',  'W', '10'),
    ('K', 'XR',  'W', '15'),
    ('K', 'XR',  'W', '20'),
    ('K', 'XR',  'W', '25'),
    ('K', 'XR',  'W', '30'),
    ('K', 'XR',  'W', '35'),
    ('K', 'XR',  'W', '40'),
    ('K', 'XR',  'W', '43'), 
    
        #12,15,20,25,30,40,50,60,70,80,90,100
    ('C', 'LR', 'D', '12'),
    ('C', 'LR', 'D', '15'),
    ('C', 'LR', 'D', '20'),
    ('C', 'LR', 'D', '25'),
    ('C', 'LR', 'D', '30'),
    ('C', 'LR', 'D', '40'),
    ('C', 'LR', 'D', '50'),
    ('C', 'LR', 'D', '60'),
    ('C', 'LR', 'D', '70'),
    ('C', 'LR', 'D', '80'),
    ('C', 'LR', 'D', '90'),
    ('C', 'LR', 'D', '100'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048 
    ('C', 'RR', 'D', '12'),
    ('C', 'RR', 'D', '50'),
    ('C', 'RR', 'D', '100'),
    ('C', 'RR', 'D', '200'),
    ('C', 'RR', 'D', '300'),
    ('C', 'RR', 'D', '400'),
    ('C', 'RR', 'D', '500'),
    ('C', 'RR', 'D', '750'),
    ('C', 'RR', 'D', '1000'),
    ('C', 'RR', 'D', '1250'),
    ('C', 'RR', 'D', '1500'),
    ('C', 'RR', 'D', '1750'),
    ('C', 'RR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'SR', 'D', '12'),
    ('C', 'SR', 'D', '50'),
    ('C', 'SR', 'D', '100'),
    ('C', 'SR', 'D', '200'),
    ('C', 'SR', 'D', '300'),
    ('C', 'SR', 'D', '400'),
    ('C', 'SR', 'D', '500'),
    ('C', 'SR', 'D', '750'),
    ('C', 'SR', 'D', '1000'),
    ('C', 'SR', 'D', '1250'),
    ('C', 'SR', 'D', '1500'),
    ('C', 'SR', 'D', '1750'),
    ('C', 'SR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'XR', 'D', '12'),
    ('C', 'XR', 'D', '50'),
    ('C', 'XR', 'D', '100'),
    ('C', 'XR', 'D', '200'),
    ('C', 'XR', 'D', '300'),
    ('C', 'XR', 'D', '400'),
    ('C', 'XR', 'D', '500'),
    ('C', 'XR', 'D', '750'),
    ('C', 'XR', 'D', '1000'),
    ('C', 'XR', 'D', '1250'),
    ('C', 'XR', 'D', '1500'),
    ('C', 'XR', 'D', '1750'),
    ('C', 'XR', 'D', '2048'),
    
    #12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    ('C', 'LR', 'W', '12'),
    ('C', 'LR', 'W', '14'),
    ('C', 'LR', 'W', '16'),
    ('C', 'LR', 'W', '18'),
    ('C', 'LR', 'W', '20'),
    ('C', 'LR', 'W', '22'),
    ('C', 'LR', 'W', '24'),
    ('C', 'LR', 'W', '26'),
    ('C', 'LR', 'W', '28'),
    ('C', 'LR', 'W', '30'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'RR', 'W', '12'),
    ('C', 'RR', 'W', '50'),
    ('C', 'RR', 'W', '100'),
    ('C', 'RR', 'W', '200'),
    ('C', 'RR', 'W', '300'),
    ('C', 'RR', 'W', '400'),
    ('C', 'RR', 'W', '1250'),
    ('C', 'RR', 'W', '1500'),
    ('C', 'RR', 'W', '1750'),
    ('C', 'RR', 'W', '2048'),
    
    

    
    #12, 20,30,40,50,60,70,80,90, 100, 115, 130, 150
    ('C', 'SR', 'W', '12'),
    ('C', 'SR', 'W', '20'),
    ('C', 'SR', 'W', '30'),
    ('C', 'SR', 'W', '40'),
    ('C', 'SR', 'W', '50'),
    ('C', 'SR', 'W', '60'),
    ('C', 'SR', 'W', '70'),
    ('C', 'SR', 'W', '80'),
    ('C', 'SR', 'W', '90'),
    ('C', 'SR', 'W', '100'),
    ('C', 'SR', 'W', '115'),
    ('C', 'SR', 'W', '130'),
    ('C', 'SR', 'W', '150'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'XR', 'W', '12'),
    ('C', 'XR', 'W', '50'),
    ('C', 'XR', 'W', '100'),
    ('C', 'XR', 'W', '200'),
    ('C', 'XR', 'W', '300'),
    ('C', 'XR', 'W', '400'),
    ('C', 'XR', 'W', '500'),
    ('C', 'XR', 'W', '750'),
    ('C', 'XR', 'W', '1000'),
    ('C', 'XR', 'W', '1250'),
    ('C', 'XR', 'W', '1500'),
    ('C', 'XR', 'W', '1750'),
    ('C', 'XR', 'W', '2048'),
    
    #12,15,20,25,30,40,50,60,70,80,90,100
    ('G', 'LR', 'D', '12'),
    ('G', 'LR', 'D', '15'),
    ('G', 'LR', 'D', '20'),
    ('G', 'LR', 'D', '25'),
    ('G', 'LR', 'D', '30'),
    ('G', 'LR', 'D', '40'),
    ('G', 'LR', 'D', '50'),
    ('G', 'LR', 'D', '60'),
    ('G', 'LR', 'D', '70'),
    ('G', 'LR', 'D', '80'),
    ('G', 'LR', 'D', '90'),
    ('G', 'LR', 'D', '100'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'RR', 'W', '500'),
    ('C', 'RR', 'W', '750'),
    ('C', 'RR', 'W', '1000'),

    ('G', 'RR', 'D', '12'),
    ('G', 'RR', 'D', '50'),
    ('G', 'RR', 'D', '100'),
    ('G', 'RR', 'D', '200'),
    ('G', 'RR', 'D', '300'),
    ('G', 'RR', 'D', '400'),
    ('G', 'RR', 'D', '500'),
    ('G', 'RR', 'D', '750'),
    ('G', 'RR', 'D', '1000'),
    ('G', 'RR', 'D', '1250'),
    ('G', 'RR', 'D', '1500'),
    ('G', 'RR', 'D', '1750'),
    ('G', 'RR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'SR', 'D', '12'),
    ('G', 'SR', 'D', '50'),
    ('G', 'SR', 'D', '100'),
    ('G', 'SR', 'D', '200'),
    ('G', 'SR', 'D', '300'),
    ('G', 'SR', 'D', '400'),
    ('G', 'SR', 'D', '500'),
    ('G', 'SR', 'D', '750'),
    ('G', 'SR', 'D', '1000'),
    ('G', 'SR', 'D', '1250'),
    ('G', 'SR', 'D', '1500'),
    ('G', 'SR', 'D', '1750'),
    ('G', 'SR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'XR', 'D', '12'),
    ('G', 'XR', 'D', '50'),
    ('G', 'XR', 'D', '100'),
    ('G', 'XR', 'D', '200'),
    ('G', 'XR', 'D', '300'),
    ('G', 'XR', 'D', '400'),
    ('G', 'XR', 'D', '500'),
    ('G', 'XR', 'D', '750'),
    ('G', 'XR', 'D', '1000'),
    ('G', 'XR', 'D', '1250'),
    ('G', 'XR', 'D', '1500'),
    ('G', 'XR', 'D', '1750'),
    ('G', 'XR', 'D', '2048'),
    
    #12, 15, 20, 25, 30, 40, 50, 60, 70, 80
    ('G', 'LR', 'W', '12'),
    ('G', 'LR', 'W', '15'),
    ('G', 'LR', 'W', '20'),
    ('G', 'LR', 'W', '25'),
    ('G', 'LR', 'W', '30'),
    ('G', 'LR', 'W', '40'),
    ('G', 'LR', 'W', '50'),
    ('G', 'LR', 'W', '60'),
    ('G', 'LR', 'W', '70'),
    ('G', 'LR', 'W', '80'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'RR', 'W', '12'),
    ('G', 'RR', 'W', '50'),
    ('G', 'RR', 'W', '100'),
    ('G', 'RR', 'W', '200'),
    ('G', 'RR', 'W', '300'),
    ('G', 'RR', 'W', '400'),
    ('G', 'RR', 'W', '500'),
    ('G', 'RR', 'W', '750'),
    ('G', 'RR', 'W', '1000'),
    ('G', 'RR', 'W', '1250'),
    ('G', 'RR', 'W', '1500'),
    ('G', 'RR', 'W', '1750'),
    ('G', 'RR', 'W', '2048'),
    
    #12, 20,30,40,50,60,70,80,90, 100, 115, 130, 150
    ('G', 'SR', 'W', '12'),
    ('G', 'SR', 'W', '20'),
    ('G', 'SR', 'W', '30'),
    ('G', 'SR', 'W', '40'),
    ('G', 'SR', 'W', '50'),
    ('G', 'SR', 'W', '60'),
    ('G', 'SR', 'W', '70'),
    ('G', 'SR', 'W', '80'),
    ('G', 'SR', 'W', '90'),
    ('G', 'SR', 'W', '100'),
    ('G', 'SR', 'W', '115'),
    ('G', 'SR', 'W', '130'),
    ('G', 'SR', 'W', '150'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'XR', 'W', '12'),
    ('G', 'XR', 'W', '50'),
    ('G', 'XR', 'W', '100'),
    ('G', 'XR', 'W', '200'),
    ('G', 'XR', 'W', '300'),
    ('G', 'XR', 'W', '400'),
    ('G', 'XR', 'W', '500'),
    ('G', 'XR', 'W', '750'),
    ('G', 'XR', 'W', '1000'),
    ('G', 'XR', 'W', '1250'),
    ('G', 'XR', 'W', '1500'),
    ('G', 'XR', 'W', '1750'),
    ('G', 'XR', 'W', '2048'),
    '''
    
    run_queue = [

        # 12, 14, 16
    ('C', 'LR', 'A', '12'),
    ('C', 'LR', 'A', '14'),
    ('C', 'LR', 'A', '16'),
    
    # 12, 24, 40, 60, 80, 100, 150, 200, 250, 300, 500, 750, 1024
    ('C', 'RR', 'A', '12'),
    ('C', 'RR', 'A', '24'),
    ('C', 'RR', 'A', '40'),
    ('C', 'RR', 'A', '60'),
    ('C', 'RR', 'A', '80'),
    ('C', 'RR', 'A', '100'),
    ('C', 'RR', 'A', '150'),
    ('C', 'RR', 'A', '200'),
    ('C', 'RR', 'A', '250'),
    ('C', 'RR', 'A', '300'),
    ('C', 'RR', 'A', '500'),
    ('C', 'RR', 'A', '750'),
    ('C', 'RR', 'A', '1024'),
    
    # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 200, 300, 500, 750, 1024
    ('C', 'SR', 'A', '10'),
    ('C', 'SR', 'A', '20'),
    ('C', 'SR', 'A', '30'),
    ('C', 'SR', 'A', '40'),
    ('C', 'SR', 'A', '50'),
    ('C', 'SR', 'A', '60'),
    ('C', 'SR', 'A', '70'),
    ('C', 'SR', 'A', '80'),
    ('C', 'SR', 'A', '90'),
    ('C', 'SR', 'A', '100'),
    ('C', 'SR', 'A', '125'),
    ('C', 'SR', 'A', '200'),
    ('C', 'SR', 'A', '300'),
    ('C', 'SR', 'A', '500'),
    ('C', 'SR', 'A', '750'),
    ('C', 'SR', 'A', '1024'),
    
    

    
    # 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140
    ('C', 'XR', 'A', '12'),
    ('C', 'XR', 'A', '20'),
    ('C', 'XR', 'A', '30'),
    ('C', 'XR', 'A', '40'),
    ('C', 'XR', 'A', '50'),
    ('C', 'XR', 'A', '60'),
    ('C', 'XR', 'A', '70'),
    ('C', 'XR', 'A', '80'),
    ('C', 'XR', 'A', '90'),
    ('C', 'XR', 'A', '100'),
    ('C', 'XR', 'A', '110'),
    ('C', 'XR', 'A', '120'),
    ('C', 'XR', 'A', '130'),
    ('C', 'XR', 'A', '140'),
    
    # 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46
    ('G', 'LR', 'A', '12'),
    ('G', 'LR', 'A', '14'),
    ('G', 'LR', 'A', '16'),
    ('G', 'LR', 'A', '18'),  
    ('G', 'LR', 'A', '22'),
    ('G', 'LR', 'A', '24'),
    ('G', 'LR', 'A', '26'),
    ('G', 'LR', 'A', '28'),
    ('G', 'LR', 'A', '30'),
    ('G', 'LR', 'A', '32'),
    ('G', 'LR', 'A', '34'),
    ('G', 'LR', 'A', '36'),
    ('G', 'LR', 'A', '38'),
    ('G', 'LR', 'A', '42'),
    ('G', 'LR', 'A', '44'),
    ('G', 'LR', 'A', '46'),
    
    # 12, 24, 40, 60, 80, 100, 150, 200, 250, 300, 500, 750, 1024
    ('G', 'RR', 'A', '12'),
    ('G', 'RR', 'A', '24'),
    ('G', 'RR', 'A', '40'),
    ('G', 'RR', 'A', '60'),
    ('G', 'RR', 'A', '80'),
    ('G', 'RR', 'A', '100'),
    ('G', 'RR', 'A', '150'),
    ('G', 'RR', 'A', '200'),
    ('G', 'RR', 'A', '250'),
    ('G', 'RR', 'A', '300'),
    ('G', 'RR', 'A', '500'),
    ('G', 'RR', 'A', '750'),
    ('G', 'RR', 'A', '1024'),
    
    # 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175
    ('G', 'SR', 'A', '12'),
    ('G', 'SR', 'A', '20'),
    ('G', 'SR', 'A', '30'),
    ('G', 'SR', 'A', '40'),
    ('G', 'SR', 'A', '60'),
    ('G', 'SR', 'A', '70'),
    ('G', 'SR', 'A', '80'),
    ('G', 'SR', 'A', '90'),
    ('G', 'SR', 'A', '100'),
    ('G', 'SR', 'A', '125'),
    ('G', 'SR', 'A', '150'),
    ('G', 'SR', 'A', '175'),
    ('G', 'SR', 'A', '500'),
    ('G', 'SR', 'A', '1024'),
    
    ('G', 'XR', 'A', '12'),
    ('G', 'XR', 'A', '20'),
    ('G', 'XR', 'A', '30'),
    ('G', 'XR', 'A', '40'),
    ('G', 'XR', 'A', '50'),
    ('G', 'XR', 'A', '60'),
    ('G', 'XR', 'A', '70'),
    ('G', 'XR', 'A', '80'),
    ('G', 'XR', 'A', '90'),
    ('G', 'XR', 'A', '100'),
    ('G', 'XR', 'A', '125'),
    ('G', 'XR', 'A', '150'),
    ('G', 'XR', 'A', '175'),
    ('G', 'XR', 'A', '300'),
    ('G', 'XR', 'A', '600'),
    ('G', 'XR', 'A', '1024'),
    

    
    # 10, 30, 34, 40, 44, 50, 54, 60, 64, 70, 74, 80
    ('K', 'LR', 'A', '10'),
    ('K', 'LR', 'A', '30'),
    ('K', 'LR', 'A', '34'),
    ('K', 'LR', 'A', '40'),
    ('K', 'LR', 'A', '44'),
    ('K', 'LR', 'A', '50'),
    ('K', 'LR', 'A', '54'),
    ('K', 'LR', 'A', '60'),
    ('K', 'LR', 'A', '64'),
    ('K', 'LR', 'A', '70'),
    ('K', 'LR', 'A', '74'),
    ('K', 'LR', 'A', '80'),
    
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    ('K', 'RR', 'A', '1'),
    ('K', 'RR', 'A', '2'),
    ('K', 'RR', 'A', '3'),
    ('K', 'RR', 'A', '4'),
    ('K', 'RR', 'A', '5'),
    ('K', 'RR', 'A', '6'),
    ('K', 'RR', 'A', '7'),
    ('K', 'RR', 'A', '8'),
    ('K', 'RR', 'A', '9'),
    ('K', 'RR', 'A', '10'),
    ('K', 'RR', 'A', '11'),
    ('K', 'RR', 'A', '12'),
    ('K', 'RR', 'A', '13'),
    ('K', 'RR', 'A', '14'),
    ('K', 'RR', 'A', '15'),
    
    # 1, 2, 3, 4, 5
    ('K', 'SR', 'A', '1'),
    ('K', 'SR', 'A', '2'),
    ('K', 'SR', 'A', '3'),
    ('K', 'SR', 'A', '4'),
    
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ('K', 'XR', 'A', '1'),
    ('K', 'XR', 'A', '2'),
    ('K', 'XR', 'A', '3'),
    ('K', 'XR', 'A', '4'),
    ('K', 'XR', 'A', '5'),
    ('K', 'XR', 'A', '6'),
    ('K', 'XR', 'A', '7'),
    ('K', 'XR', 'A', '8'),
    ('K', 'XR', 'A', '9'),
    ('K', 'XR', 'A', '10'),
    ('K', 'XR', 'A', '11'),
    ('K', 'XR', 'A', '12'),
    ('K', 'XR', 'A', '13'),
    ('K', 'XR', 'A', '14'),
    ('K', 'XR', 'A', '15'),
    ('K', 'XR', 'A', '16'),
    ('K', 'XR', 'A', '17'),
    ('K', 'XR', 'A', '18'),
    ('K', 'XR', 'A', '19'),
    
    # 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700
    ('S', 'LR', 'A', '25'),
    ('S', 'LR', 'A', '50'),
    ('S', 'LR', 'A', '75'),
    ('S', 'LR', 'A', '100'),
    ('S', 'LR', 'A', '150'),
    ('S', 'LR', 'A', '200'),
    ('S', 'LR', 'A', '250'),
    ('S', 'LR', 'A', '300'),
    ('S', 'LR', 'A', '400'),
    ('S', 'LR', 'A', '500'),
    ('S', 'LR', 'A', '600'),
    ('S', 'LR', 'A', '700'),
    
    # 10, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000
    ('S', 'RR', 'A', '50'),
    ('S', 'RR', 'A', '100'),
    ('S', 'RR', 'A', '200'),
    ('S', 'RR', 'A', '300'),
    ('S', 'RR', 'A', '400'),
    ('S', 'RR', 'A', '500'),
    ('S', 'RR', 'A', '750'),
    ('S', 'RR', 'A', '1000'),
    ('S', 'RR', 'A', '1250'),
    ('S', 'RR', 'A', '1500'),
    ('S', 'RR', 'A', '1750'),
    ('S', 'RR', 'A', '2000'),
    ('S', 'RR', 'A', '2500'),
    ('S', 'RR', 'A', '3000'),
    ('S', 'RR', 'A', '3500'),
    ('S', 'RR', 'A', '4000'),
    
    # 20, 30, 50, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 2000
    ('S', 'SR', 'A', '20'),
    ('S', 'SR', 'A', '30'),
    ('S', 'SR', 'A', '50'),
    ('S', 'SR', 'A', '80'),
    ('S', 'SR', 'A', '100'),
    ('S', 'SR', 'A', '150'),
    ('S', 'SR', 'A', '200'),
    ('S', 'SR', 'A', '300'),
    ('S', 'SR', 'A', '400'),
    ('S', 'SR', 'A', '500'),
    ('S', 'SR', 'A', '600'),
    ('S', 'SR', 'A', '700'),
    ('S', 'SR', 'A', '800'),
    ('S', 'SR', 'A', '900'),
    ('S', 'SR', 'A', '1000'),
    ('S', 'SR', 'A', '1250'),
    ('S', 'SR', 'A', '1500'),
    ('S', 'SR', 'A', '2000'),
    
    # 20, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000, 6000, 7000
    ('S', 'XR', 'A', '20'),
    ('S', 'XR', 'A', '50'),
    ('S', 'XR', 'A', '100'),
    ('S', 'XR', 'A', '200'),
    ('S', 'XR', 'A', '300'),
    ('S', 'XR', 'A', '400'),
    ('S', 'XR', 'A', '500'),
    ('S', 'XR', 'A', '750'),
    ('S', 'XR', 'A', '1000'),
    ('S', 'XR', 'A', '1500'),
    ('S', 'XR', 'A', '2000'),
    ('S', 'XR', 'A', '2500'),
    ('S', 'XR', 'A', '3000'),
    ('S', 'XR', 'A', '4000'),
    ('S', 'XR', 'A', '5000'),
    ('S', 'XR', 'A', '6000'),
    ('S', 'XR', 'A', '7000'),
    
      #1, 10, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000
    ('K', 'LR', 'M', '1'),
    ('K', 'LR', 'M', '10'),
    ('K', 'LR', 'M', '50'),
    ('K', 'LR', 'M', '100'),
    ('K', 'LR', 'M', '200'),
    ('K', 'LR', 'M', '300'),
    ('K', 'LR', 'M', '400'),
    ('K', 'LR', 'M', '500'),
    ('K', 'LR', 'M', '1000'),
    ('K', 'LR', 'M', '2000'),
    ('K', 'LR', 'M', '3000'),
    ('K', 'LR', 'M', '4000'),
    ('K', 'LR', 'M', '5000'),
    ('K', 'LR', 'M', '6000'),
    ('K', 'LR', 'M', '7000'),
    ('K', 'LR', 'M', '8000'),
    
    #1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250 
    ('K', 'RR', 'M', '1'),
    ('K', 'RR', 'M', '5'),
    ('K', 'RR', 'M', '10'),
    ('K', 'RR', 'M', '20'),
    ('K', 'RR', 'M', '30'),
    ('K', 'RR', 'M', '40'),
    ('K', 'RR', 'M', '50'),
    ('K', 'RR', 'M', '75'),
    ('K', 'RR', 'M', '100'),
    ('K', 'RR', 'M', '150'),
    ('K', 'RR', 'M', '200'),
    ('K', 'RR', 'M', '250'),
    
    #1, 2, 3, 4, 5, 6, 8, 10, 12,
    ('K', 'SR', 'M', '1'),
    ('K', 'SR', 'M', '2'),
    ('K', 'SR', 'M', '3'),
    ('K', 'SR', 'M', '4'),
    ('K', 'SR', 'M', '5'),
    ('K', 'SR', 'M', '6'),
    ('K', 'SR', 'M', '8'),
    ('K', 'SR', 'M', '10'),
    ('K', 'SR', 'M', '12'),
    
    #1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500
    ('K', 'XR', 'M', '1'),
    ('K', 'XR', 'M', '5'),
    ('K', 'XR', 'M', '10'),
    ('K', 'XR', 'M', '25'),
    ('K', 'XR', 'M', '50'),
    ('K', 'XR', 'M', '75'),
    ('K', 'XR', 'M', '100'),
    ('K', 'XR', 'M', '200'),
    ('K', 'XR', 'M', '300'),
    ('K', 'XR', 'M', '400'),
    ('K', 'XR', 'M', '500'),
    ('K', 'XR', 'M', '600'),

    
    #30, 100, 200, 300, 400, 500, 800, 1400, 1500, 3000, 5000, 10000, 20000, 30000, 40000, 50000
    ('S', 'LR', 'M', '30'),
    ('S', 'LR', 'M', '100'),
    ('S', 'LR', 'M', '200'),
    ('S', 'LR', 'M', '300'),
    ('S', 'LR', 'M', '400'),
    ('S', 'LR', 'M', '500'),
    ('S', 'LR', 'M', '800'),
    ('S', 'LR', 'M', '1400'),
    ('S', 'LR', 'M', '1500'),
    ('S', 'LR', 'M', '3000'),
    ('S', 'LR', 'M', '5000'),
    ('S', 'LR', 'M', '10000'),
    ('S', 'LR', 'M', '20000'),
    ('S', 'LR', 'M', '30000'),
    ('S', 'LR', 'M', '40000'),
    ('S', 'LR', 'M', '50000'),
    
    #30, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000
    ('S', 'RR', 'M', '30'),
    ('S', 'RR', 'M', '100'),
    ('S', 'RR', 'M', '200'),
    ('S', 'RR', 'M', '500'),
    ('S', 'RR', 'M', '1000'),
    ('S', 'RR', 'M', '1500'),
    ('S', 'RR', 'M', '2000'),
    ('S', 'RR', 'M', '3000'),
    ('S', 'RR', 'M', '4000'),
    ('S', 'RR', 'M', '5000'),
    ('S', 'RR', 'M', '6000'),
    ('S', 'RR', 'M', '7000'),
    ('S', 'RR', 'M', '8000'),
    ('S', 'RR', 'M', '9000'),
    ('S', 'RR', 'M', '10000'),
    ('S', 'RR', 'M', '12000'),
    
    #30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 6000
    ('S', 'SR', 'M', '30'),
    ('S', 'SR', 'M', '40'),
    ('S', 'SR', 'M', '50'),
    ('S', 'SR', 'M', '75'),
    ('S', 'SR', 'M', '100'),
    ('S', 'SR', 'M', '150'),
    ('S', 'SR', 'M', '200'),
    ('S', 'SR', 'M', '300'),
    ('S', 'SR', 'M', '500'),
    ('S', 'SR', 'M', '750'),
    ('S', 'SR', 'M', '1000'),
    ('S', 'SR', 'M', '1250'),
    ('S', 'SR', 'M', '1500'),
    ('S', 'SR', 'M', '2000'),
    ('S', 'SR', 'M', '2500'),
    ('S', 'SR', 'M', '3000'),
    ('S', 'SR', 'M', '4000'),
    ('S', 'SR', 'M', '5000'),
    ('S', 'SR', 'M', '6000'),
    
    
    #30, 50, 100, 400, 1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000
    ('S', 'XR', 'M', '150'),
    ('S', 'XR', 'M', '400'),
    ('S', 'XR', 'M', '1000'),
    ('S', 'XR', 'M', '3000'),
    ('S', 'XR', 'M', '5000'),
    ('S', 'XR', 'M', '10000'),
    ('S', 'XR', 'M', '15000'),
    ('S', 'XR', 'M', '20000'),
    ('S', 'XR', 'M', '25000'),
    ('S', 'XR', 'M', '30000'),
    ('S', 'XR', 'M', '35000'),
    ('S', 'XR', 'M', '40000'),
    ('S', 'XR', 'M', '45000'),
    ('S', 'XR', 'M', '50000'),
    

    #16, 50, 100, 200, 300, 400, 1000, 3000, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 32768
    ('C', 'LR', 'M', '16'),
    ('C', 'LR', 'M', '50'),
    ('C', 'LR', 'M', '100'),
    ('C', 'LR', 'M', '200'),
    ('C', 'LR', 'M', '300'),
    ('C', 'LR', 'M', '400'),
    ('C', 'LR', 'M', '1000'),
    ('C', 'LR', 'M', '3000'),
    ('C', 'LR', 'M', '5000'),
    ('C', 'LR', 'M', '7500'),
    ('C', 'LR', 'M', '10000'),
    ('C', 'LR', 'M', '12500'),
    ('C', 'LR', 'M', '15000'),
    ('C', 'LR', 'M', '17500'),
    ('C', 'LR', 'M', '20000'),
    ('C', 'LR', 'M', '25000'),
    ('C', 'LR', 'M', '32768'),
    
    
    #15, 50, 70, 90, 110, 150, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 32768
    ('C', 'RR', 'M', '16'),
    ('C', 'RR', 'M', '50'),
    ('C', 'RR', 'M', '70'),
    ('C', 'RR', 'M', '90'),
    ('C', 'RR', 'M', '110'),
    ('C', 'RR', 'M', '150'),
    ('C', 'RR', 'M', '500'),
    ('C', 'RR', 'M', '1000'),
    ('C', 'RR', 'M', '2000'),
    ('C', 'RR', 'M', '3000'),
    ('C', 'RR', 'M', '5000'),
    ('C', 'RR', 'M', '7500'),
    ('C', 'RR', 'M', '10000'),
    ('C', 'RR', 'M', '15000'),
    ('C', 'RR', 'M', '20000'),
    ('C', 'RR', 'M', '25000'),
    ('C', 'RR', 'M', '32768'),
    
    #16, 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 32768
    ('C', 'SR', 'M', '16'),
    ('C', 'SR', 'M', '50'),
    ('C', 'SR', 'M', '100'),
    ('C', 'SR', 'M', '250'),
    ('C', 'SR', 'M', '500'),
    ('C', 'SR', 'M', '1000'),
    ('C', 'SR', 'M', '2500'),
    ('C', 'SR', 'M', '5000'),
    ('C', 'SR', 'M', '7500'),
    ('C', 'SR', 'M', '10000'),
    ('C', 'SR', 'M', '12500'),
    ('C', 'SR', 'M', '15000'),
    ('C', 'SR', 'M', '20000'),
    ('C', 'SR', 'M', '25000'),
    ('C', 'SR', 'M', '32768'),
    
    #16, 50, 150, 300, 500, 700, 1000, 1250, 1500, 2000, 3000, 4000, 5000, 7500, 10000
    ('C', 'XR', 'M', '16'),
    ('C', 'XR', 'M', '50'),
    ('C', 'XR', 'M', '150'),
    ('C', 'XR', 'M', '300'),
    ('C', 'XR', 'M', '500'),
    ('C', 'XR', 'M', '700'),
    ('C', 'XR', 'M', '1000'),
    ('C', 'XR', 'M', '1250'),
    ('C', 'XR', 'M', '1500'),
    ('C', 'XR', 'M', '2000'),
    ('C', 'XR', 'M', '3000'),
    ('C', 'XR', 'M', '4000'),
    ('C', 'XR', 'M', '5000'),
    ('C', 'XR', 'M', '7500'),
    ('C', 'XR', 'M', '10000'),
    
 
    
    #16, 50, 100, 250, 500, 1000, 2000, 5000, 7500, 10000, 15000, 20000, 25000, 32768
    ('G', 'LR', 'M', '16'), 
    ('G', 'LR', 'M', '50'),
    ('G', 'LR', 'M', '100'),
    ('G', 'LR', 'M', '250'),
    ('G', 'LR', 'M', '500'),
    ('G', 'LR', 'M', '1000'),
    ('G', 'LR', 'M', '2000'),
    ('G', 'LR', 'M', '5000'),
    ('G', 'LR', 'M', '7500'),
    ('G', 'LR', 'M', '10000'),
    ('G', 'LR', 'M', '15000'),
    ('G', 'LR', 'M', '20000'),
    ('G', 'LR', 'M', '25000'),
    ('G', 'LR', 'M', '32768'),
 
    
    # 16, 50, 60, 80, 100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 32768
    ('G', 'RR', 'M', '16'),
    ('G', 'RR', 'M', '50'),
    ('G', 'RR', 'M', '60'),
    ('G', 'RR', 'M', '80'),
    ('G', 'RR', 'M', '100'),
    ('G', 'RR', 'M', '500'),
    ('G', 'RR', 'M', '1000'),
    ('G', 'RR', 'M', '2000'),
    ('G', 'RR', 'M', '3000'),
    ('G', 'RR', 'M', '5000'),
    ('G', 'RR', 'M', '7500'),
    ('G', 'RR', 'M', '10000'),
    ('G', 'RR', 'M', '15000'),
    ('G', 'RR', 'M', '20000'),
    ('G', 'RR', 'M', '25000'),
    ('G', 'RR', 'M', '32768'),

    
    #16, 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 32768
    ('G', 'SR', 'M', '16'),
    ('G', 'SR', 'M', '50'),
    ('G', 'SR', 'M', '100'),
    ('G', 'SR', 'M', '250'),
    ('G', 'SR', 'M', '500'),
    ('G', 'SR', 'M', '1000'),
    ('G', 'SR', 'M', '2500'),
    ('G', 'SR', 'M', '5000'),
    ('G', 'SR', 'M', '7500'),
    ('G', 'SR', 'M', '10000'),
    ('G', 'SR', 'M', '12500'),
    ('G', 'SR', 'M', '15000'),
    ('G', 'SR', 'M', '20000'),
    ('G', 'SR', 'M', '25000'),
    ('G', 'SR', 'M', '32768'),
    
    # 16, 50, 70, 90, 150, 300, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000
    ('G', 'XR', 'M', '16'),
    ('G', 'XR', 'M', '50'),
    ('G', 'XR', 'M', '70'),
    ('G', 'XR', 'M', '90'),
    ('G', 'XR', 'M', '150'),
    ('G', 'XR', 'M', '300'),
    ('G', 'XR', 'M', '500'),
    ('G', 'XR', 'M', '1000'),
    ('G', 'XR', 'M', '2500'),
    ('G', 'XR', 'M', '5000'),
    ('G', 'XR', 'M', '7500'),
    ('G', 'XR', 'M', '10000'),
    ('G', 'XR', 'M', '12500'),
    ('G', 'XR', 'M', '15000'),


    
    
    ]
    


    for approximation, model, data, n_samples in run_queue:
        running_counter += 1
        key = f": M={model}, A={approximation}, D={data}, N={n_samples}"
        running_command_stability = [
            "python", "main.py",
            "--mode", "E",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(n_samples),
            "--eval_method", "R",
        ]
        
        try:
            __write_result_to_file_and_print(f"Started RBO: {key} - Number({running_counter})")
            subprocess.run(running_command_stability, check=True, cwd=src)
            __write_result_to_file_and_print(f"Finished")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: = {str(e)} ({running_counter})")
    
    
    for approximation, model, data, n_samples in run_queue:
        key = f": M={model}, A={approximation}, D={data}, N={n_samples}"
        running_command_fidelity = [
            "python", "main.py",
            "--mode", "E",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(n_samples),
            "--eval_method", "F"
        ]


        
        try:
            __write_result_to_file_and_print(f"Started Fidelity: {key} ({running_counter})")
            subprocess.run(running_command_fidelity, check=True, cwd=src)
            __write_result_to_file_and_print(f"Finished")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: {key} = {str(e)} ({running_counter})")
            
            
def run_stability():
    
    running_counter = 0
    src = str(src_dir)


    '''
    #D & LR - 3,5s
    ("C","D","LR","12"),
    ("C","D","LR","20"),

    ("G","D","LR","12"),
    ("G","D","LR","120"),

    ("K","D","LR","800"),
    ("K","D","LR","1700"),

    ("S","D","LR","1400"),
    ("S","D","LR","2300"),

    #D & RR - 3,6,10s Lower Bound (S) > Upper Bound (K,C)
    ("C","D","RR","20"),
    ("C","D","RR","120"),
    # not possible

    ("G","D","RR","250"),
    ("G","D","RR","130"),
    # not possible

    ("K","D","RR","4"),
    ("K","D","RR","25"),
    ("K","D","RR","52"),

    # not possible
    # not possible
    ("S","D","RR","25"),


    #D & SR - 10,25s
    ("C","D","SR","120"),
    ("C","D","SR","1300"),

    ("G","D","SR","120"),
    ("G","D","SR","270"),

    ("K","D","SR","9"),
    ("K","D","SR","40"),

    ("S","D","SR","1000"),
    ("S","D","SR","3000"),


    #D & XR - 3,8s
    ("C","D","XR","20"),
    ("C","D","XR","220"),

    ("G","D","XR","46"),
    ("G","D","XR","2048"),

    ("K","D","XR","35"),
    ("K","D","XR","120"),

    ("S","D","XR","1500"),
    ("S","D","XR","4300"),

    #W & LR - 15,30s
    ("C","W","LR","12"),
    ("C","W","LR","26"),

    ("G","W","LR","36"),

    ("K","W","LR","28"),
    ("K","W","LR","70"),

    ("S","W","LR","500"),
    ("S","W","LR","1150"),

    #W & RR - 20, 80s, 130s
    ("C","W","RR","12"),
    ("C","W","RR","46"), 
    ("C","W","RR","86"),

    ("G","W","RR","40"),
    ("G","W","RR","2048"),
    # not possible

    # not possible
    ("K","W","RR","1"),
    ("K","W","RR","2"),
    
        # not possible
    # not possible
    ("S","W","RR","30"),
    
    #W & SR - 75s, 100
    ("C","W","SR","12"),
    ("C","W","SR","16"),

    ("G","W","SR","12"),
    ("G","W","SR","18"),

    # not possible
    # not possible

    ("S","W","SR","50"),
    ("S","W","SR","80"),


    #W & XR - 20,50s
    ("C","W","XR","12"),
    ("C","W","XR","30"),

    ("G","W","XR","46"),
    ("G","W","XR","150"),

    ("K","W","XR","1"),
    ("K","W","XR","3"),

    ("S","W","XR","600"),
    ("S","W","XR","1500"),



    '''
    #50s and 100s
    run_queue = [
    ("S","W","SR","70"),#70
    ("S","W","SR","62"),#70
    ("S","W","SR","64"),#70
    ("S","W","SR","66"),#70
    ("S","W","SR","68"),#70
    ("S","W","SR","74"),#70
    ("S","W","SR","78"),#70
    ]

    

    for approximation, data, model, n_samples in run_queue:

        running_counter += 1
        key = f"Stability: M={model}, A={approximation}, D={data}, N={n_samples}, R=10"
        running_command_stability = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(n_samples),
            "--number_of_runs", "10"
        ]
        
        eval_command_stability = [
            "python", "main.py",
            "--mode", "E",
            "--data", data,
            "--model", model,
            "--approximation", approximation,
            "--nsamples", str(n_samples),
            "--number_of_runs", "10",
            "--eval_method", "S"
        ]

        try:
            __write_result_to_file_and_print(f"Started: {key} ({running_counter})")
            subprocess.run(running_command_stability, check=True, cwd=src)
            __write_result_to_file_and_print(f"Finished: {key} ({running_counter})")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: {key} = {str(e)} ({running_counter})")
            
        try:
            __write_result_to_file_and_print(f"Started: {key} ({running_counter})")
            subprocess.run(eval_command_stability, check=True, cwd=src)
            __write_result_to_file_and_print(f"Finished: {key} ({running_counter})")
        except Exception as e:
            __write_result_to_file_and_print(f"Error for: {key} = {str(e)} ({running_counter})")
            
            
    