from evaluation_results_loader import get_evaluation
import matplotlib.pyplot as plt



stability_runs =[
#D & LR - 3,5s
    #("C","D","LR","12"),#70
    #("C","D","LR","20"),#70
    
    ("C","D","LR","70"),#36
    ("C","D","LR","140"),#70
    ("C","D","LR","200"),

    #("G","D","LR","12"),#26
    ("G","D","LR","56"),#26
    ("G","D","LR","120"),
    ("G","D","LR","200"),

    #("K","D","LR","800"),#500
    ("K","D","LR","500"),
    #("K","D","LR","1700"),#800
    ("K","D","LR","800"),#800
    ("K","D","LR","1200"),


    ("S","D","LR","1400"),
    #("S","D","LR","2300"),#2100
    ("S","D","LR","2100"),#2100
    ("S","D","LR","2600"),
    

    #D & RR - 3,6,10s Lower Bound (S) > Upper Bound (K,C)
    ("C","D","RR","20"),
    ("C","D","RR","120"),
    ("C","D","RR","300"),
    ("C","D","RR","500"),

    # not possible
    ("G","D","RR","50"),
    ("G","D","RR","250"),
    ("G","D","RR","130"),
    ("G","D","RR","500"),
    ("G","D","RR","750"),
    # not possible

    ("K","D","RR","1"),
    ("K","D","RR","4"),
    ("K","D","RR","25"),
    ("K","D","RR","52"),
    ("K","D","RR","1"),
    ("K","D","RR","100"),

    # not possible
    # not possible
    ("S","D","RR","25"),
    ("S","D","RR","50"),


    #D & SR - 10,25s
    ("C","D","SR","120"),
    ("C","D","SR","1300"),
    ("C","D","SR","2700"),
    ("C","D","SR","5000"),

    ("G","D","SR","120"),
    ("G","D","SR","270"),
    ("G","D","SR","600"),
    ("G","D","SR","1300"),

    ("K","D","SR","9"),
    ("K","D","SR","20"),
    ("K","D","SR","40"),
    ("K","D","SR","80"),


    ("S","D","SR","1000"),
    ("S","D","SR","3000"),
    ("S","D","SR","6000"),
    ("S","D","SR","13000"),


    #D & XR - 3,8s
    ("C","D","XR","20"),
    ("C","D","XR","220"),
    ("C","D","XR","500"),
    ("C","D","XR","750"),

    ("G","D","XR","46"),
    ("G","D","XR","2048"),
    #Upper

    ("K","D","XR","35"),
    ("K","D","XR","120"),
    ("K","D","XR","1000"),
    ("K","D","XR","1600"),

    ("S","D","XR","1500"),
    ("S","D","XR","4300"),
    ("S","D","XR","10000"),
    ("S","D","XR","16000"),

    #W & LR - 15,30s
    ("C","W","LR","12"),
    ("C","W","LR","20"),

    ("G","W","LR","36"),
    ("G","W","LR","94"),

    ("K","W","LR","28"),
    ("K","W","LR","70"),

    ("S","W","LR","500"),
    ("S","W","LR","1150"),

    #W & RR - 20, 80s, 130s
    ("C","W","RR","12"),
    ("C","W","RR","46"), 
    ("C","W","RR","86"),
    ("C","W","RR","100"), 
    ("C","W","RR","300"),
    ("C","W","RR","500"),
    ("C","W","RR","1000"),

    ("G","W","RR","40"),
    ("G","W","RR","2048"),
    # not possible

    # not possible
    ("K","W","RR","1"),
    ("K","W","RR","2"),
    ("K","W","RR","3"),
    ("K","W","RR","5"),
    ("K","W","RR","10"),
    ("K","W","RR","20"),
    
    # not possible
    # not possible
    ("S","W","RR","30"),
    ("S","W","RR","50"),
    ("S","W","RR","100"),
    ("S","W","RR","500"),
    ("S","W","RR","1000"),
    
    #W & SR - 75s, 100
    ("C","W","SR","12"),
    ("C","W","SR","16"),
    ("C","W","SR","20"),

    ("G","W","SR","12"),
    ("G","W","SR","18"),
    ("G","W","SR","25"),

    # not possible
    # not possible

    ("S","W","SR","40"),
    ("S","W","SR","74"),
    ("S","W","SR","100"),

    #W & XR - 20,50s
    ("C","W","XR","12"),
    ("C","W","XR","30"),
    ("C","W","XR","50"),
    ("C","W","XR","100"),
    ("C","W","XR","200"),

    ("G","W","XR","46"),
    ("G","W","XR","150"),
    ("G","W","XR","300"),
    ("G","W","XR","500"),
    ("G","W","XR","800"),

    ("K","W","XR","1"),
    ("K","W","XR","3"),
    ("K","W","XR","5"),
    ("K","W","XR","10"),
    ("K","W","XR","20"),

    ("S","W","XR","600"),
    ("S","W","XR","1500"),
    ("S","W","XR","3000"),
    ("S","W","XR","5000"),
    ("S","W","XR","10000"),

    
    
    # A+LR [11.1 - 11.8 - 12.89]    -> C12 - C16, G12 - G46, K10 - K80, S25 - S700
    ("C","A","LR","12"),
    ("G","A","LR","38"),
    ("K","A","LR","62"),
    ("S","A","LR","250"),
    ("C","A","LR","14"),
    ("G","A","LR","44"),
    ("S","A","LR","280"),
    ("C","A","LR","16"),
    ("G","A","LR","46"),
    ("S","A","LR","320"),
    
    
    ("G","A","LR","60"),
    
    ("K","A","LR","10"),
    ("K","A","LR","20"),
    ("K","A","LR","30"),
    ("K","A","LR","40"),
    
    ("S","A","LR","400"),
    ("S","A","LR","500"),
    ("S","A","LR","600"),
    
    ("S","A","LR","50"),
    ("S","A","LR","100"),
    ("S","A","LR","300"),
    ("S","A","LR","800"),
    
    ("K","A","LR","1"),
    ("K","A","LR","5"),
    
    
    ("G","A","LR","30"),
    ("G","A","LR","24"),
    ("G","A","LR","18"),
    ("G","A","LR","12"),
    
    # A + RR [76, 102] -> C12 - C1024, G12 - G1024, K1 - K15, S50 - S4000
    ("C","A","RR", "80"),
    ("G","A","RR", "130"),
    ("K","A","RR", "2"),
    ("S","A","RR", "50"),
    ("C","A","RR", "90"),
    ("G","A","RR", "140"),
    ("K","A","RR", "3"),
    ("S","A","RR", "100"),
    ("C","A","RR", "100"),
    ("G","A","RR", "150"),
    ("K","A","RR", "4"),
    ("S","A","RR", "400"),
    ("C","A","RR", "500"),
    ("G","A","RR", "160"),
    ("K","A","RR", "5"),
    ("S","A","RR", "750"),
    
    ("C","A","RR", "60"),
    ("C","A","RR", "200"),
    ("C","A","RR", "300"),
    
    ("G","A","RR", "200"),
    ("G","A","RR", "300"),
    ("G","A","RR", "400"),
    ("G","A","RR", "500"),
    ("G","A","RR", "600"),
    
    ("K","A","RR", "6"),
    ("K","A","RR", "7"),
    ("K","A","RR", "8"),
    ("K","A","RR", "9"),
    ("K","A","RR", "10"),
    
    ("S","A","RR", "30"),
    ("S","A","RR", "900"),
    ("S","A","RR", "1000"),
    ("S","A","RR", "1250"),
    ("S","A","RR", "15000"),
    
    ("K","A","RR", "1"),
    ("K","A","RR", "15"),
    ("S","A","RR", "20000"),
    ("S","A","RR", "50000"),
    ("G","A","RR", "12"),
    ("G","A","RR", "50"),
    ("G","A","RR", "800"),
    ("G","A","RR", "1024"),
    ("C","A","RR", "50"),
    ("C","A","RR", "30"),
    ("C","A","RR", "12"),
    ("C","A","RR", "500"),
    ("C","A","RR", "800"),
    ("C","A","RR", "1024"),
    
    # A + XR [17 - 102 ] 
    ("C","A","XR", "30"),
    ("C","A","XR", "48"),
    ("C","A","XR", "62"),
    ("C","A","XR", "76"),
    ("C","A","XR", "84"),
    ("C","A","XR", "104"),
    ("C","A","XR", "114"),
    ("C","A","XR", "122"),
    ("C","A","XR", "134"),
      
    ("G","A","XR", "80"),
    ("G","A","XR", "100"),
    ("G","A","XR", "110"),
    ("G","A","XR", "124"),
    ("G","A","XR", "134"),
    ("G","A","XR", "144"),
    ("G","A","XR", "200"),
    ("G","A","XR", "300"),

    
    ("K","A","XR", "3"),
    ("K","A","XR", "4"),
    ("K","A","XR", "5"),
    ("K","A","XR", "9"),
    ("K","A","XR", "10"),
    ("K","A","XR", "12"),
    ("K","A","XR", "14"),
    ("K","A","XR", "16"),
    ("K","A","XR", "18"),
    
    
    ("S","A","XR", "1000"),
    ("S","A","XR", "1500"),
    ("S","A","XR", "2000"),
    ("S","A","XR", "3000"),
    ("S","A","XR", "3500"),
    ("S","A","XR", "4000"),
    ("S","A","XR", "4500"),
    ("S","A","XR", "5300"),
    ("S","A","XR", "6000"),
    
    #A + SR ->[33, 730]
    ("S","A","SR", "60"),
    ("S","A","SR", "100"),
    ("S","A","SR", "120"),
    ("S","A","SR", "150"),
    
    ("C","A","SR", "12"),
    ("C","A","SR", "20"),
    ("C","A","SR", "26"),
    ("C","A","SR", "34"),
    
    ("G","A","SR", "16"),
    ("G","A","SR", "22"),
    ("G","A","SR", "30"),
    ("G","A","SR", "36"),
    
    ("K","A","SR", "1"),
    #("K","A","SR", "2"),
    
    ("G","A","SR", "360"),
    ("C","A","SR", "340"),
    ("S","A","SR", "1500"),
    
    
    # M + LR -> [6 - 20]
    ('C', 'M', 'LR', '16'),
    ('C', 'M', 'LR', '50'),
    ('C', 'M', 'LR', '90'),
    ('C', 'M', 'LR', '120'),
    ('S', 'M', 'LR', '800'),
    ('S', 'M', 'LR', '1400'),
    ('S', 'M', 'LR', '2000'),
    ('S', 'M', 'LR', '2600'),
    ('G', 'M', 'LR', '40'),
    ('G', 'M', 'LR', '90'),
    ('G', 'M', 'LR', '160'),
    ('G', 'M', 'LR', '250'),
    ('K', 'M', 'LR', '150'),
    ('K', 'M', 'LR', '400'),
    ('K', 'M', 'LR', '500'),
    ('K', 'M', 'LR', '800'),
    
    ('C', 'M', 'LR', '200'),
    ('C', 'M', 'LR', '300'),
    ('C', 'M', 'LR', '400'),
    ('C', 'M', 'LR', '600'),
    ('S', 'M', 'LR', '3200'),
    ('S', 'M', 'LR', '4000'),
    ('S', 'M', 'LR', '8000'),
    ('S', 'M', 'LR', '12000'),
    ('G', 'M', 'LR', '400'),
    ('G', 'M', 'LR', '800'),
    ('G', 'M', 'LR', '1200'),
    ('G', 'M', 'LR', '1500'),
    ('K', 'M', 'LR', '1500'),
    ('K', 'M', 'LR', '2000'),
    ('K', 'M', 'LR', '3000'),
    ('K', 'M', 'LR', '5000'),

    # M + RR
    ('G', 'M', 'RR', '80'),
    ('G', 'M', 'RR', '300'),
    ('C', 'M', 'RR', '50'),
    ('C', 'M', 'RR', '120'),
    ('C', 'M', 'RR', '32768'),
    ('K', 'M', 'RR', '400'),
    ('K', 'M', 'RR', '900'),
    ('S', 'M', 'RR', '1400'),
    ('S', 'M', 'RR', '1800'),
    ('S', 'M', 'RR', '5000'),
    
    
    ('K', 'M', 'RR', '50'),
    ('K', 'M', 'RR', '100'),
    ('K', 'M', 'RR', '200'),
    ('G', 'M', 'RR', '1000'),
    ('G', 'M', 'RR', '10000'),
    ('G', 'M', 'RR', '32768'),
    ('S', 'M', 'RR', '10000'),
    ('S', 'M', 'RR', '20000'),
    ('S', 'M', 'RR', '30000'),
    ('S', 'M', 'RR', '40000'),
    ('S', 'M', 'RR', '50000'),

    # M + XR
    ('G', 'M', 'XR', '16'),
    ('G', 'M', 'XR', '150'),
    ('G', 'M', 'XR', '300'),
    ('C', 'M', 'XR', '24'),
    ('C', 'M', 'XR', '90'),
    ('C', 'M', 'XR', '170'),
    ('S', 'M', 'XR', '400'),
    ('S', 'M', 'XR', '1500'),
    ('S', 'M', 'XR', '2200'),
    ('S', 'M', 'XR', '4000'),
    ('K', 'M', 'XR', '10'),
    ('K', 'M', 'XR', '30'),
    ('K', 'M', 'XR', '50'),
    ('K', 'M', 'XR', '60'),
    
    
    ('G', 'M', 'XR', '600'),
    ('G', 'M', 'XR', '1500'),
    ('G', 'M', 'XR', '7000'),
    ('C', 'M', 'XR', '400'),
    ('C', 'M', 'XR', '900'),
    ('C', 'M', 'XR', '2000'),
    ('S', 'M', 'XR', '10000'),
    ('S', 'M', 'XR', '20000'),
    ('S', 'M', 'XR', '50000'),
    ('K', 'M', 'XR', '150'),
    ('K', 'M', 'XR', '300'),
    ('K', 'M', 'XR', '600'),

    # M + SR
    ('K', 'M', 'SR', '1'),
    ('K', 'M', 'SR', '2'),
    ('K', 'M', 'SR', '3'),
    ('S', 'M', 'SR', '400'),
    ('S', 'M', 'SR', '750'),
    ('S', 'M', 'SR', '1000'),
    ('C', 'M', 'SR', '60'),
    ('C', 'M', 'SR', '110'),
    ('C', 'M', 'SR', '150'),
    ('G', 'M', 'SR', '80'),
    ('G', 'M', 'SR', '120'),
    ('G', 'M', 'SR', '15000'),
    
    ('K', 'M', 'SR', '6'),
    ('K', 'M', 'SR', '10'),
    ('K', 'M', 'SR', '40'),
    ('S', 'M', 'SR', '2000'),
    ('S', 'M', 'SR', '4000'),
    ('S', 'M', 'SR', '12000'),
    ('C', 'M', 'SR', '300'),
    ('C', 'M', 'SR', '600'),
    ('C', 'M', 'SR', '2000'),
    ('G', 'M', 'SR', '32000'),
    ]

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

df_A = df_stability[df_stability["index"].str.contains("_A_")]
df_A_dict = dict()
df_A_dict_colors = dict()
exact_shap_values_A = {"LR":13.3, "RR":255,"SR":16180.786,"XR":105}

df_M = df_stability[df_stability["index"].str.contains("_M_")]
df_M_dict = dict()
df_M_dict_colors = dict()
exact_shap_values_M = {"LR":298.284, "RR":4673.568,"SR":122133.64,"XR":2255.235}

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
        model_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["Stability_Score"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Diabetes Dataset")
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
        model_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["Stability_Score"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Diabetes Dataset")
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
        model_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["Stability_Score"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_D_dict_colors[key], linestyle='-')
    
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Diabetes Dataset")
plt.legend(title="Approximation", loc="best")
plt.show()
'''

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
        model_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["Stability_Score"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Diabetes Dataset")
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
        model_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["Stability_Score"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Wine Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=19.373605728149414, color='blue', linestyle=':', linewidth=1, label="Copula SHAP - Sample Upper Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best", fontsize=6)
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
        model_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["Stability_Score"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Wine Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=78.775649, color='green', linestyle='-', linewidth=1, label="Gaussian SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=123.42659, color='purple', linestyle=':', linewidth=1, label="Sample SHAP - Sample Lower Bound", alpha=0.6)
plt.axvline(x=103.550527, color='orange', linestyle=':', linewidth=1, label="Kernel SHAP - Sample Lower Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#W - SR
models = ["SR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_"))]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        model_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["Stability_Score"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Wine Dataset")
plt.legend(title="Approximation", loc="best")
plt.show()'''

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
        model_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["Stability_Score"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Wine Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#A - LR
models = ["LR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_A[(df_A["index"].str.contains(f"{approximation}_")) & (df_A["index"].str.contains(f"_{model}_"))]
        df_A_dict[f"{approximation}_{model}"]=df
        df_A_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_A[model]
        model_name = model_names[model]
        
for key in df_A_dict.keys():
    plt.scatter(df_A_dict[key]["Running_Time"], df_A_dict[key]["Stability_Score"], color=df_A_dict_colors[key], label=shap_names[key[0]])
    df = df_A_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_A_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Abalone Dataset")
plt.axvline(x=11.269, color='blue', linestyle=':', linewidth=1, label="Sample SHAP - Copula Lower Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#A - RR
models = ["RR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_A[(df_A["index"].str.contains(f"{approximation}_")) & (df_A["index"].str.contains(f"_{model}_"))]
        df_A_dict[f"{approximation}_{model}"]=df
        df_A_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_A[model]
        model_name = model_names[model]
        
for key in df_A_dict.keys():
    plt.scatter(df_A_dict[key]["Running_Time"], df_A_dict[key]["Stability_Score"], color=df_A_dict_colors[key], label=shap_names[key[0]])
    df = df_A_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_A_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Abalone Dataset")
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#A - SR
models = ["SR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_A[(df_A["index"].str.contains(f"{approximation}_")) & (df_A["index"].str.contains(f"_{model}_"))]
        df_A_dict[f"{approximation}_{model}"]=df
        df_A_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        model_name = model_names[model]
        
for key in df_A_dict.keys():
    plt.scatter(df_A_dict[key]["Running_Time"], df_A_dict[key]["Stability_Score"], color=df_A_dict_colors[key], label=shap_names[key[0]])
    df = df_A_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_A_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Abalone Dataset")
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#A - XR
models = ["XR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_A[(df_A["index"].str.contains(f"{approximation}_")) & (df_A["index"].str.contains(f"_{model}_"))]
        df_A_dict[f"{approximation}_{model}"]=df
        df_A_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_A[model]
        model_name = model_names[model]
        
for key in df_A_dict.keys():
    plt.scatter(df_A_dict[key]["Running_Time"], df_A_dict[key]["Stability_Score"], color=df_A_dict_colors[key], label=shap_names[key[0]])
    df = df_A_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_A_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Abalone Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#M - LR
models = ["LR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_M[(df_M["index"].str.contains(f"{approximation}_")) & (df_M["index"].str.contains(f"_{model}_"))]
        df_M_dict[f"{approximation}_{model}"]=df
        df_M_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_M[model]
        model_name = model_names[model]
        
for key in df_M_dict.keys():
    plt.scatter(df_M_dict[key]["Running_Time"], df_M_dict[key]["Stability_Score"], color=df_M_dict_colors[key], label=shap_names[key[0]])
    df = df_M_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_M_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Music Dataset")
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#M - RR
models = ["RR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_M[(df_M["index"].str.contains(f"{approximation}_")) & (df_M["index"].str.contains(f"_{model}_"))]
        df_M_dict[f"{approximation}_{model}"]=df
        df_M_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_M[model]
        model_name = model_names[model]
        
for key in df_M_dict.keys():
    plt.scatter(df_M_dict[key]["Running_Time"], df_M_dict[key]["Stability_Score"], color=df_M_dict_colors[key], label=shap_names[key[0]])
    df = df_M_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_M_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Music Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#M - XR
models = ["XR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_M[(df_M["index"].str.contains(f"{approximation}_")) & (df_M["index"].str.contains(f"_{model}_"))]
        df_M_dict[f"{approximation}_{model}"]=df
        df_M_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        #exact_shap_value = exact_shap_values_M[model]
        model_name = model_names[model]
        
for key in df_M_dict.keys():
    plt.scatter(df_M_dict[key]["Running_Time"], df_M_dict[key]["Stability_Score"], color=df_M_dict_colors[key], label=shap_names[key[0]])
    df = df_M_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_M_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
#plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Music Dataset")
plt.legend(title="Approximation", loc="best")
plt.show()'''


#M - SR
models = ["SR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_M[(df_M["index"].str.contains(f"{approximation}_")) & (df_M["index"].str.contains(f"_{model}_"))]
        df_M_dict[f"{approximation}_{model}"]=df
        df_M_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_M[model]
        model_name = model_names[model]
        
for key in df_M_dict.keys():
    plt.scatter(df_M_dict[key]["Running_Time"], df_M_dict[key]["Stability_Score"], color=df_M_dict_colors[key], label=shap_names[key[0]])
    df = df_M_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["Stability_Score"], color=df_M_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in Seconds")
plt.ylabel("Stability Score")
#plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.title(f"Stability Comparison (10 runs) - {model_name} & Music Dataset")
plt.legend(title="Approximation", loc="best")
plt.show()