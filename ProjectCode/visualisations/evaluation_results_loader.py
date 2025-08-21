import sys
import os
import numpy as np
import pickle
from pathlib import Path

project_start_point = Path(__file__).resolve().parent.parent
src_path = project_start_point / "src"
sys.path.append(str(src_path))

from shap_approximations.shap_loader import get_shap_values
from evaluations.result_organizer import get_results
from models.model_loader import get_model
os.chdir(project_start_point / "src")


def get_evaluation (eval_method: list = None):
    df = get_results()
    if eval_method is not None:
        df = df[eval_method]
    
    return df

def _load_shap_pickle(approx, data_name, model_name, nsamples=None):
    """Lädt das SHAP-Pickle und gibt (shap_vals_2d, X_test, feature_names, path) zurück."""
    fname = f"{approx}_{data_name}_{model_name}_{'None' if approx=='E' else nsamples}.pkl"
    path = Path("shap_approximations/shap_values") / fname
    with open(path, "rb") as f:
        d = pickle.load(f)

    shap_vals = d["shap_values"]
    if hasattr(shap_vals, "values"):  # falls shap.Explanation gespeichert
        shap_vals = shap_vals.values
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals[None, :]
    X_test = np.asarray(d["X_test"])
    feature_names = list(d.get("feature_names", []))
    return shap_vals, X_test, feature_names, path

def recover_base_from_pickle(data_name, model_name, approx, nsamples=None):
    """
    Berechnet den Base-Vektor für die angegebene Approximation:
    base_per_instance = model.predict(X_test) - shap_vals.sum(axis=1)
    """
    shap_vals, X_test, _, path = _load_shap_pickle(approx, data_name, model_name, nsamples)
    model = get_model(data_name, model_name)
    yhat = np.asarray(model.predict(X_test)).reshape(-1)  # Regression: (n,)
    if yhat.shape[0] != shap_vals.shape[0]:
        raise ValueError(f"[{approx}] yhat n={yhat.shape[0]} passt nicht zu shap n={shap_vals.shape[0]} (Datei: {path.name})")
    base_per_instance = yhat - shap_vals.sum(axis=1)
    return base_per_instance  # shape: (n,)

def recover_base_from_exact(data_name, model_name):
    """Deine alte Funktion – bleibt als Komfort für Exact."""
    return (
        float(recover_base_from_pickle(data_name, model_name, "E").mean()),
        float(recover_base_from_pickle(data_name, model_name, "E").std()),
        recover_base_from_pickle(data_name, model_name, "E"),
    )
