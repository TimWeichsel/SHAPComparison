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

    
    
def plot_shap_rank_comparison(data_name_long, model_name,
                               C_samples, G_samples, K_samples, S_samples, seconds,
                               instance_index=None, eval_method="RBO_Score_0.9",
                               figsize=(18, 6), MC_100 = False):
    
    # get first letter
    data_name = data_name_long[0].upper()

    # --- Original Column Names holen ---
    data = get_data(data_name, 0)
    column_names = data["X_train"].columns.tolist()

    # Projektpfad setzen
    project_start_point = Path(__file__).resolve().parent.parent
    os.chdir(project_start_point / "src")

    # Methoden definieren mit schöneren Namen
    
    
    if MC_100 is False:
        methods = [
            ("E", None, "Exact SHAP"),
            ("C", C_samples, "Copula SHAP (10 MC)"),
            ("G", G_samples, "Gaussian SHAP (10 MC)"),
            ("K", K_samples, "Kernel SHAP"),
            ("S", S_samples, "Sample SHAP"),
        ]
    else:
        methods = [
            ("E", None, "Exact SHAP"),
            ("C100", C_samples, "Copula SHAP (100 MC)"),
            ("G100", G_samples, "Gaussian SHAP (100 MC)"),
            ("K", K_samples, "Kernel SHAP"),
            ("S", S_samples, "Sample SHAP"),
        ]

    shap_dict = {}

    # ---- SHAP-Werte laden ----
    for approx, ns, _ in methods:
        shap_vals = np.asarray(get_shap_values(
            data_name=data_name,
            model_name=model_name,
            approximation_name=approx,
            nsamples=ns,
            instance_index=None  # alles holen
        ))

        if shap_vals.ndim == 1:
            shap_vals = shap_vals[None, :]

        if instance_index is None:
            vals = shap_vals.mean(axis=0)
        else:
            vals = shap_vals[instance_index]

        shap_dict[approx] = vals.ravel()

    # ---- DataFrame bauen ----
    shap_df = pd.DataFrame(shap_dict, index=column_names)

    # ---- Farben ----
    base_colors = [
        "blue", "orange", "green", "red", "purple", "brown", "gray",
        "olive", "magenta", "gold", "teal", "navy", "maroon", "lime", "coral"
    ]
    colors = {column_names[i]: base_colors[i % len(column_names)]
              for i in range(len(column_names))}

    # ---- Results laden ----
    results_df = get_results()

    # ---- Plot ----
    fig, axes = plt.subplots(1, len(methods), figsize=figsize, sharex=True)

    for ax, (approx, ns, title) in zip(axes, methods):
        sorted_feats = shap_df[approx].abs().sort_values(ascending=False).index
        vals = shap_df.loc[sorted_feats, approx]

        # Metrik nur für Approx (nicht E)
        eval_value = None
        if approx != "E" and eval_method in results_df.columns:
            key_prefix = f"{approx}_{data_name}_{model_name}_{ns}"
            if key_prefix in results_df.index:
                eval_value = results_df.loc[key_prefix, eval_method]
                # einzelne Instanz oder Durchschnitt
                if isinstance(eval_value, (list, np.ndarray)) and instance_index is not None:
                    eval_value = eval_value[instance_index]
                elif isinstance(eval_value, (list, np.ndarray)):
                    eval_value = np.nanmean(eval_value)

        # Plot Balken
        ax.barh(range(len(vals)), vals, color=[colors[f] for f in sorted_feats])

        # passendes Label in den Titel schreiben
        if eval_value is not None and not np.isnan(eval_value):
            if eval_method == "RBO_Score_0.9":
                ax.set_title(f"{title} – RBO: {eval_value:.3f}", fontsize=10)
            elif eval_method == "Fidelity_Score_MAE":
                ax.set_title(f"{title} – MAE: {eval_value:.3f}", fontsize=10)
            else:
                ax.set_title(title, fontsize=10)
        else:
            ax.set_title(title, fontsize=10)

        ax.set_yticks(range(len(sorted_feats)))
        ax.set_yticklabels(sorted_feats, fontsize=6)
        ax.axvline(0, color="black", linewidth=1)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.tight_layout(rect=[0.00, 0.0, 1, 0.95])

    if instance_index is None:
        sup_title = f"Average SHAP Values for {data_name_long} - {model_name} (Running Time ≈ {seconds} seconds)"
    else:
        sup_title = f"SHAP Values of Instance {instance_index} for {data_name_long} - {model_name} (Running Time ≈ {seconds} seconds)"

    plt.suptitle(sup_title, fontsize=14)
    plt.show()


if __name__ == "__main__":
    
   #LR
    plot_shap_rank_comparison(
        data_name_long="Diabetes", model_name="LR",
        C_samples=160, G_samples=260, K_samples=10000, S_samples=2500,
        seconds=6,
        instance_index=None,   
        eval_method="RBO_Score_0.9",  # oder "Fidelity_Score_MAE"
        figsize=(15, 6)
    )
    
    plot_shap_rank_comparison(
        data_name_long="Music", model_name="LR",
        C_samples=50, G_samples=50, K_samples=8000, S_samples=50000,
        seconds=300,
        instance_index=None,   
        eval_method="Fidelity_Score_MAE", 
        figsize=(15, 6), MC_100=True
    )

    
    # RR
    plot_shap_rank_comparison(
        data_name_long="Abalone", model_name="RR",
        C_samples=12, G_samples=12, K_samples=14, S_samples=4000,
        seconds=250,
        instance_index=None,   
        eval_method="RBO_Score_0.9", 
        figsize=(15, 6), MC_100=True
    )

    plot_shap_rank_comparison(
        data_name_long="Diabetes", model_name="RR",
        C_samples=24, G_samples=24, K_samples=300, S_samples=12000,
        seconds=50,
        instance_index=None,   
        eval_method="Fidelity_Score_MAE",  # oder "RBO_Score_0.9"
        figsize=(15, 6), MC_100=True
    )

    
    #SR

    plot_shap_rank_comparison(
        data_name_long="Abalone", model_name="SR",
        C_samples=24, G_samples=24, K_samples=7, S_samples=10000, #Determine right Samples for K and S
        seconds=6000,
        instance_index=None,   
        eval_method="Fidelity_Score_MAE",  # oder "Fidelity_Score_MAE"
        figsize=(15, 6), MC_100=True
    )

    
    plot_shap_rank_comparison(
        data_name_long="Music", model_name="SR",
        C_samples=24, G_samples=24, K_samples=30, S_samples=10000,
        seconds=500,
        instance_index=None,   
        eval_method="RBO_Score_0.9",  # oder 
        figsize=(15, 6), MC_100=True
    )
    
    #Backup
    '''
    plot_shap_rank_comparison(
        data_name_long="Wine", model_name="SR",
        C_samples=12, G_samples=12, K_samples=2, S_samples=5000,
        seconds=6000,
        instance_index=None,   
        eval_method="Fidelity_Score_MAE",  # oder "RBO_Score_0.9"
        figsize=(15, 6), MC_100=True
    )'''
    
    
    #XR -> 2. Updaten und auf RBO/MSA entscheiden 
    
    plot_shap_rank_comparison(
        data_name_long="Abalone", model_name="XR",
        C_samples=140, G_samples=175, K_samples=18, S_samples=6000,
        seconds=100,
        instance_index=None,   
        eval_method="RBO_Score_0.9",  # oder "Fidelity_Score_MAE"
        figsize=(15, 6)
    )
    

    plot_shap_rank_comparison(
        data_name_long="Wine", model_name="XR",
        C_samples=50, G_samples=1000 , K_samples=5, S_samples=3000,
        seconds=80,
        instance_index=None,   
        eval_method="Fidelity_Score_MAE",  # oder "Fidelity_Score_MAE"
        figsize=(15, 6)
    )
    
    
#Backup
'''
    plot_shap_rank_comparison(
        data_name_long="Wine", model_name="XR",
        C_samples=18, G_samples=18 , K_samples=35, S_samples=20000,
        seconds=500,
        instance_index=None,   
        eval_method="RBO_Score_0.9",  # oder "Fidelity_Score_MAE"
        figsize=(15, 6), MC_100=True
    )'''