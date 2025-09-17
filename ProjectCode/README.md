# 1 A Comparative Evaluation of Model-Agnostic SHAP Approximations
Bachelor Thesis Project by BSc. Tim Weichselberger (University of Vienna) supervised by Prof. Ass.-Prof. Dipl.-Ing. Dr.techn. Sebastian Tschiatschek

This thesis' project investigates how well standard model-agnostic SHAP approximations align with the evaluation properties of Degree of Importance (RBO), Fidelity(MAE) and Stability(Confidence Interval) across different machine learning models.

# 2 Requirements
- Python (3.11.12 was used)
- R (4.3.1)
- Git

# 3 Installation

## (a) Package Installation (Python)
requirements.txt includes all needed python packages for this installation (for R packages, see next point)

```bash 
pip install -r requirements.txt
```

## (b) Installation of shapr and R packages (R libraries)
This project is using the shapr implementation for the Gaussian and Copula SHAP approximation. To use shapr the Python wrapper shaprpy is needed.

To install shapr for this project run the script:
```bash 
bash setup_shapr.sh
```

# 4 Start Project 
```bash 
python src/main.py --help
```

# 5 Usage
The main file has multiple modes: train models, compute SHAP values, evaluate SHAP values, load SHAP values and print results. All the necessary information is provided in the main.py file and can be seen when the code of "Start Project" is run.