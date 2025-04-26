
# Installation

## Package Installation
requirements.txt includes all needed packages for this installation (except shapr, see next point)

```bash 
pip install -r requirements.txt
```

## Installation of shapr
This project is using the shapr implementation for the Gaussian and Copula SHAP approximation. To use shapr the Python wrapper shaprpy is needed.

To install shapr for this project run the script:
```bash 
setup_shapr.sh
```