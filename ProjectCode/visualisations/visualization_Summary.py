import shap
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from evaluation_results_loader import recover_base_from_exact

project_start_point = Path(__file__).resolve().parent.parent
os.chdir(project_start_point / "src")


import shap
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from shap_approximations.shap_loader import get_shap_values
from data.data_loader import get_data
from evaluation_results_loader import get_results

import shap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Projektpfad setzen
project_start_point = Path(__file__).resolve().parent.parent
os.chdir(project_start_point / "src")

# Eigene Loader
from shap_approximations.shap_loader import get_shap_values
from data.data_loader import get_data

def summary_plot_all_methods_auto(
    data_name_long: str,
    model_name: str,
    nsamples_dict: dict | None = None,
    max_display: int = 15,
    runtime: float | dict | None = None,
    show_logs: bool = True,
    plot_size=(2, 2.6),      # << NEU: schmale Figure (Breite, Höhe in Zoll)
    left=0.28, right=0.98,     # << NEU: Ränder enger
    share_xlim=True            # << NEU: einheitliche x-Achse für alle Methoden
):
    import numpy as np
    import shap
    import matplotlib.pyplot as plt

    data_name = data_name_long[0].upper()
    data = get_data(data_name, 0)
    X_train, X_test = data["X_train"], data["X_test"]

    if nsamples_dict is None:
        nsamples_dict = {"E": None, "K": 20, "S": 1000, "G": 12, "C": 12}

    methods = {
        "Exact SHAP":    ("E", nsamples_dict.get("E")),
        "Kernel SHAP":   ("K", nsamples_dict.get("K")),
        "Sample SHAP":   ("S", nsamples_dict.get("S")),
        "Gaussian SHAP": ("G", nsamples_dict.get("G")),
        "Copula SHAP":   ("C", nsamples_dict.get("C")),
    }

    # Optional: globale x-Achsen-Grenze bestimmen (99%-Quantil vermeidet Ausreißer-Overkill)
    x_abs_max = None
    if share_xlim:
        tmp_max = []
        for _, (approx, ns) in methods.items():
            sh = np.asarray(get_shap_values(
                data_name=data_name, model_name=model_name,
                approximation_name=approx, nsamples=ns, instance_index=None
            ))
            if sh.ndim == 1: sh = sh[None, :]
            tmp_max.append(np.nanpercentile(np.abs(sh), 99))
        x_abs_max = float(np.nanmax(tmp_max))

    for nice_name, (approx, ns) in methods.items():
        shap_vals = np.asarray(get_shap_values(
            data_name=data_name,
            model_name=model_name,
            approximation_name=approx,
            nsamples=ns,
            instance_index=None
        ))
        if shap_vals.ndim == 1:
            shap_vals = shap_vals[None, :]

        n_rows, n_feats = shap_vals.shape

        if X_test.shape[0] == n_rows:
            X_base = X_test
        elif X_train.shape[0] == n_rows:
            X_base = X_train
        else:
            if show_logs:
                print(f"⚠️ {nice_name}: {shap_vals.shape} passt weder zu X_train {X_train.shape} noch X_test {X_test.shape}.")
            continue

        if X_base.shape[1] != n_feats:
            if show_logs:
                print(f"⚠️ {nice_name}: Feature-Dimension ungleich ({n_feats} vs {X_base.shape[1]}).")
            continue

        if show_logs:
            print(f"✓ {nice_name}: benutze Shape {X_base.shape} (SHAP {shap_vals.shape})")

        # Nur bei Sample SHAP die Farbskala
        color_bar = (approx == "S")

        # --- SCHMALER PLOT ---
        shap.summary_plot(
            shap_vals,
            X_base,
            feature_names=X_base.columns,
            plot_type="dot",
            max_display=max_display,
            show=False,
            color_bar=color_bar,
            plot_size=plot_size    # << das macht die Figure wirklich schmaler
        )

        ax = plt.gca()
        # Ränder enger
        plt.gcf().subplots_adjust(left=left, right=right)

        # Einheitliche x-Achse (optional, gut gegen „Luft“)
        if x_abs_max is not None and np.isfinite(x_abs_max):
            ax.set_xlim(-x_abs_max, x_abs_max)

        # Kleinere Tick-Labels für kompaktere Breite
        ax.tick_params(axis='both', labelsize=8)

        plt.title(f"{nice_name}", fontsize=10)
        plt.tight_layout()
        plt.show()
        plt.close()



if __name__ == "__main__":
    '''
    summary_plot_all_methods_auto(
        data_name_long="Abalone",
        model_name="RR",
        nsamples_dict={"E": None, "K": 14, "S": 4000, "G": 12, "C": 12},
        max_display=15,
        runtime=250  # gilt für alle Methoden
    )'''
    
    summary_plot_all_methods_auto(
        data_name_long="Wine",
        model_name="XR",
        nsamples_dict={"E": None, "K": 20, "S": 10000, "G": 12, "C": 12},
        max_display=15,
        runtime=300 
    )
    