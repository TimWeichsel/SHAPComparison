from evaluation_results_loader import get_evaluation
import matplotlib.pyplot as plt



stability_runs =[
#D & LR - 3,5s
    #("C","D","LR","12"),#70
    #("C","D","LR","20"),#70
    
    ("C","D","LR","70"),#36
    ("C","D","LR","140"),#70

    #("G","D","LR","12"),#26
    ("G","D","LR","56"),#26
    ("G","D","LR","120"),

    #("K","D","LR","800"),#500
    ("K","D","LR","500"),
    #("K","D","LR","1700"),#800
    ("K","D","LR","800"),#800

    ("S","D","LR","1400"),
    #("S","D","LR","2300"),#2100
    ("S","D","LR","2100"),#2100
    

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
    #("C","W","LR","26"),#20
    ("C","W","LR","20"),

    ("G","W","LR","36"),
    #("G","W","LR","80"),#94
    ("G","W","LR","94"),

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

    #("S","W","SR","50"),#40
    ("S","W","SR","40"),
    #("S","W","SR","80"),#70
    ("S","W","SR","74"),#70

    #W & XR - 20,50s
    ("C","W","XR","12"),
    ("C","W","XR","30"),

    ("G","W","XR","46"),
    ("G","W","XR","150"),

    ("K","W","XR","1"),
    ("K","W","XR","3"),

    ("S","W","XR","600"),
    ("S","W","XR","1500")]

approx_keys = ["_".join(key) for key in stability_runs]
approx_keys_runs = [f"{key}_10" for key in approx_keys]

df = get_evaluation().reset_index()
df_stability = df[df["index"].isin(approx_keys_runs)]

df_D = df_stability[df_stability["index"].str.contains("_D_")]
df_D_dict = dict()
df_D_dict_colors = dict()
exact_shap_values_D = {"LR":6.731802940368652, "RR":49.28979778289795,"SR":419.9395990371704,"XR":30.09837508201599}

df_W = df_stability[df_stability["index"].str.contains("_W_")]
df_W_dict = dict()
df_W_dict_colors = dict()
exact_shap_values_W = {"LR":30.49867606163025, "RR":1666.557268857956,"SR":153204.29993009567,"XR":576.2753148078918}

approximations = ["C", "G", "K", "S"]
colors = {"C":"blue", "G":"green","K":"orange","S":"purple"}

model_names = {"LR":"Linear Regression", "RR":"Random Forest","SR":"SVM","XR":"XGBoost"}
shap_names = {"C":"Copula SHAP", "G":"Gaussian SHAP","K":"Kernel SHAP","S":"Sample SHAP"}


'''
#D - LR
models = ["LR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_D[(df_D["index"].str.contains(f"{approximation}_")) & (df_D["index"].str.contains(f"_{model}_"))]
        df_D_dict[f"{approximation}_{model}"]=df
        df_D_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_D[model]
        shap_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["Stability_Score"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {shap_name} Diabetes Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#D - RR
models = ["RR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_D[(df_D["index"].str.contains(f"{approximation}_")) & (df_D["index"].str.contains(f"_{model}_"))]
        df_D_dict[f"{approximation}_{model}"]=df
        df_D_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_D[model]
        shap_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["Stability_Score"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {shap_name} Diabetes Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=9, color='blue', linestyle=':', linewidth=1, label="Copula SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=7.5, color='green', linestyle='-', linewidth=1, label="Gaussian SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=10, color='purple', linestyle='-', linewidth=1, label="Sample SHAP - Sample Lower Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#D - SR
models = ["SR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_D[(df_D["index"].str.contains(f"{approximation}_")) & (df_D["index"].str.contains(f"_{model}_"))]
        df_D_dict[f"{approximation}_{model}"]=df
        df_D_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_D[model]
        shap_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["Stability_Score"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {shap_name} Diabetes Dataset")
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#D - XR
models = ["XR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_D[(df_D["index"].str.contains(f"{approximation}_")) & (df_D["index"].str.contains(f"_{model}_"))]
        df_D_dict[f"{approximation}_{model}"]=df
        df_D_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_D[model]
        shap_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["Stability_Score"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {shap_name} Diabetes Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=8, color='blue', linestyle=':', linewidth=1, label="Copula SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=7, color='green', linestyle='-', linewidth=1, label="Gaussian SHAP - Sample Upper Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''


'''
#W - LR
models = ["LR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_"))]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_W[model]
        shap_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["Stability_Score"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {shap_name} Wine Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#W - RR
models = ["RR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_"))]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_W[model]
        shap_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["Stability_Score"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {shap_name} Wine Dataset")
#plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.legend(title="Approximation", loc="best")
plt.show()'''


#W - SR
models = ["SR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_"))]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_W[model]
        shap_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["Stability_Score"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {shap_name} Wine Dataset")
#plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.legend(title="Approximation", loc="best")
plt.show()
'''
#W - XR
models = ["XR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_"))]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_W[model]
        shap_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["Stability_Score"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {shap_name} Wine Dataset")
#plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.legend(title="Approximation", loc="best")
plt.show()'''