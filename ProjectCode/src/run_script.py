import subprocess
import itertools

# Nur Regression-Modelle
regression_models = ["LR", "RR", "SR", "XR"]

# Beide Datensätze
datasets = ["A", "W"]

# Alle Approximations außer 'B' (Bounded SHAP)
approximations = ["K", "S", "C", "G", "E"]

# nsamples-Werte
nsamples_values = [10, 100, 1000]

# Für jede Kombination aus Data, Model, Approx, nsamples
for data, model, approx, nsamples in itertools.product(datasets, regression_models, approximations, nsamples_values):
    key = f"D={data}, M={model}, A={approx}, N={nsamples}"
    command = [
        "python", "main.py",
        "--mode", "R",
        "--data", data,
        "--model", model,
        "--approximation", approx,
        "--nsamples", str(nsamples)
    ]
    
    try:
        print(f"⏳ Running: {key}")
        subprocess.run(command, check=True)
        print(f"✅ Success: {key}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler bei: {key}")
        print(f"    Fehlercode: {e.returncode}")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler bei: {key}")
        print(f"    {str(e)}")
