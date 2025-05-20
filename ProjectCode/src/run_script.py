import subprocess
import itertools

#Regression Models
regression_models = ["LR", "RR", "SR", "XR"]

#Classification Models
classification_models = ["LO", "RC", "SC", "XC"]

#Datasets Regression
reg_data = ["A", "W"]

#Datasets Classification
classification_data = ["W"]

#All Approximation-Methods
approximations = ["K", "S", "C", "G", "E"]

#nsamples values
nsamples_values = [10, 100, 1000]

running_id = 0
#All Combinations of datasets, regression models, approximations and nsamples
print("Running Process Regression started")
for data, model, approximation, nsamples in itertools.product(reg_data, regression_models, approximations, nsamples_values): #iterating over all possible combinations
    running_id += 1
    if approximation == "E":
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
        print(f"Started: {key} ({running_id})")
        subprocess.run(running_command, check=True)
        print(f"Finished: {key} ({running_id})")
    except Exception as e:
        print(f"Error for: {key} = {str(e)} ({running_id})")

#All Combinations of datasets, regression models, approximations and nsamples
print("Running Process Classification started")
for data, model, approximation, nsamples in itertools.product(classification_data, classification_models, approximations, nsamples_values): #iterating over all possible combinations
    running_id += 1
    if approximation == "E":
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
        print(f"Started: {key}")
        subprocess.run(running_command, check=True)
        print(f"Finished: {key}")
    except Exception as e:
        print(f"Error for: {key} = {str(e)}")
        
print("Running Processes ended")