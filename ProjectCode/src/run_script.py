import subprocess
import itertools

#Regression Models
regression_models = ["LR", "RR", "SR", "XR"]

#Classification Models
classification_models = ["LO", "RC", "SC", "XC"]

#All Datasets
datasets = ["A", "W"]

#All Approximation-Methods
approximations = ["K", "S", "C", "G", "E"]

#nsamples values
nsamples_values = [10, 100, 1000]

#All Combinations of datasets, regression models, approximations and nsamples
print("Running Process started")
for data, model, approx, nsamples in itertools.product(datasets, regression_models, approximations, nsamples_values):
    if approx == "E":
        key = f"D={data}, M={model}, A={approx}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approx,
        ]
    else: 
        key = f"D={data}, M={model}, A={approx}, N={nsamples}"
        running_command = [
            "python", "main.py",
            "--mode", "R",
            "--data", data,
            "--model", model,
            "--approximation", approx,
            "--nsamples", str(nsamples)
        ]
    
    try:
        print(f"Started: {key}")
        subprocess.run(running_command, check=True)
        print(f"Finished: {key}")
    except Exception as e:
        print(f"Error for: {key} = {str(e)}")
print("Running Process ended")