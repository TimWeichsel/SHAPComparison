from evaluation_results_loader import get_evaluation
import matplotlib.pyplot as plt


model = "SR"
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
    ("C","D","RR","1000"),
    ("C","D","RR","1500"),
    ("C","D","RR","2048"),

    # not possible
    ("G","D","RR","50"),
    ("G","D","RR","130"),
    ("G","D","RR","250"),

    ("G","D","RR","500"),
    ("G","D","RR","750"),
    ("G","D","RR","1000"),
    ("G","D","RR","1500"),
    ("G","D","RR","2048"),
    # not possible

    ("K","D","RR","1"),
    ("K","D","RR","4"),
    ("K","D","RR","25"),
    ("K","D","RR","52"),
    ("K","D","RR","1"),
    ("K","D","RR","100"),
    ("K","D","RR","200"),
    ("K","D","RR","300"),
    ("K","D","RR","500"),

    # not possible
    # not possible
    ("S","D","RR","25"),
    ("S","D","RR","50"),
    ("S","D","RR","100"),
    ("S","D","RR","150"),
    ("S","D","RR","500"),
    ("S","D","RR","3000"),
    ("S","D","RR","12000"),



    #D & SR - 10,25s
    ("C","D","SR","120"),
    ("C","D","SR","1300"),
    ("C","D","SR","2700"), #=2048

    ("G","D","SR","120"),
    ("G","D","SR","270"),
    ("G","D","SR","600"),
    ("G","D","SR","1300"),
    ("G","D","SR","2048"),

    ("K","D","SR","9"),
    ("K","D","SR","20"),
    ("K","D","SR","40"),
    ("K","D","SR","80"),
    ("K","D","SR","160"),
    ("K","D","SR","240"),
    ("K","D","SR","320"),

    ("S","D","SR","1000"),
    ("S","D","SR","3000"),
    ("S","D","SR","6000"),
    ("S","D","SR","13000"),
    ("S","D","SR","26000"),
    ("S","D","SR","40000"),
    ("S","D","SR","65000"),


    #D & XR - 3,8s
    ("C","D","XR","20"),
    ("C","D","XR","220"),
    ("C","D","XR","500"),
    ("C","D","XR","750"),
    ("C","D","XR","1200"),
    ("C","D","XR","2048"),

    ("G","D","XR","46"),
    ("G","D","XR","2048"),

    ("K","D","XR","35"),
    ("K","D","XR","120"),
    ("K","D","XR","500"),

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
    ("G","W","RR","500"),
    ("G","W","RR","2048"),
    # not possible

    # not possible
    ("K","W","RR","1"),
    ("K","W","RR","2"),
    ("K","W","RR","3"),
    ("K","W","RR","5"),
    ("K","W","RR","10"),
    
    # not possible
    # not possible
    ("S","W","RR","30"),
    ("S","W","RR","50"),
    ("S","W","RR","100"),
    ("S","W","RR","500"),
    ("S","W","RR","1000"),
    ("S","W","RR","1600"),
    
    #W & SR - 75s, 100
    ("C","W","SR","12"),
    ("C","W","SR","16"),
    ("C","W","SR","20"),
    ("C","W","SR","40"),
    ("C","W","SR","60"),

    ("G","W","SR","12"),
    ("G","W","SR","18"),
    ("G","W","SR","24"),
    ("G","W","SR","50"),
    ("G","W","SR","74"),

    # not possible
    # not possible

    ("S","W","SR","40"),
    ("S","W","SR","74"),
    ("S","W","SR","100"),
    ("S","W","SR","200"),
    ("S","W","SR","300"),


    #W & XR - 20,50s
    ("C","W","XR","12"),
    ("C","W","XR","30"),
    ("C","W","XR","50"),
    ("C","W","XR","100"),
    ("C","W","XR","200"),

    ("G","W","XR","46"),
    ("G","W","XR","150"),
    ("G","W","XR","300"),
    ("G","W","XR","800"),
    ("G","W","XR","1400"),#=1024
    ("G","W","XR","500"),

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
        

    ("G","A","LR","12"),
    ("C","A","LR","12"),
    ("C","A","LR","14"),
    ("C","A","LR","16"),
    ("G","A","LR","18"),
    ("G","A","LR","24"),
    ("G","A","LR","30"),
    ("G","A","LR","38"),
    ("G","A","LR","44"),
    ("G","A","LR","46"),
    ("G","A","LR","60"),
    ("K","A","LR","62"),






    
    ("S","A","LR","250"),
    ("S","A","LR","280"),
    ("S","A","LR","320"),
    
    

    
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
    ("S","A","RR", "2000"),
    ("S","A","RR", "3500"),
    
    ("K","A","RR", "1"),
    ("K","A","RR", "15"),
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
    ('C', 'M', 'LR', '1000'),
    ('C', 'M', 'LR', '2000'),
    ('C', 'M', 'LR', '3000'),
    ('S', 'M', 'LR', '3200'),
    ('S', 'M', 'LR', '4000'),
    ('S', 'M', 'LR', '8000'),
    #('S', 'M', 'LR', '12000'),
    ('G', 'M', 'LR', '400'),
    ('G', 'M', 'LR', '800'),
    ('G', 'M', 'LR', '1200'),
    ('G', 'M', 'LR', '1500'),
    ('G', 'M', 'LR', '2000'),
    ('G', 'M', 'LR', '2500'),
    ('K', 'M', 'LR', '1500'),
    #('K', 'M', 'LR', '2000'),
    #('K', 'M', 'LR', '3000'),
    #('K', 'M', 'LR', '5000'),

    # M + RR
    ('K', 'M', 'RR', '5'),
    ('S', 'M', 'RR', '100'),
    ('G', 'M', 'RR', '80'),
    ('G', 'M', 'RR', '300'),
    ('C', 'M', 'RR', '50'),
    ('C', 'M', 'RR', '120'),
    ('C', 'M', 'RR', '32768'),
    ('K', 'M', 'RR', '200'),
    ('K', 'M', 'RR', '300'),
    ('K', 'M', 'RR', '400'),
    ('S', 'M', 'RR', '1400'),
    ('S', 'M', 'RR', '1800'),
    ('S', 'M', 'RR', '5000'),
    
    
    ('K', 'M', 'RR', '50'),
    ('K', 'M', 'RR', '100'),
    ('G', 'M', 'RR', '1000'),
    ('G', 'M', 'RR', '10000'),
    ('G', 'M', 'RR', '32768'),
    ('S', 'M', 'RR', '10000'),
    ('S', 'M', 'RR', '20000'),
    ('S', 'M', 'RR', '30000'),
    ('S', 'M', 'RR', '40000'),
    ('S', 'M', 'RR', '50000'),
    ('S', 'M', 'RR', '70000'),

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
    ('G', 'M', 'XR', '32768'),
    ('C', 'M', 'XR', '32768'),   
    
    
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
    ('K', 'M', 'SR', '20'),
    ('K', 'M', 'SR', '30'),
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


# Settings
approximations = ["C", "G", "K", "S"]
colors = {"C": "blue", "G": "green", "K": "orange", "S": "purple"}
model_names = {"LR": "Linear Regression", "RR": "Random Forest", "SR": "SVM", "XR": "XGBoost"}
shap_names = {"C": "Copula SHAP", "G": "Gaussian SHAP", "K": "Kernel SHAP", "S": "Sample SHAP"}

# --- Exact SHAP Anzeigen (pro Modell & Dataset) ---
show_exact_shap_LR = {"D": True, "W": True, "A": True, "M": False}
show_exact_shap_RR = {"D": True, "W": False, "A": True, "M": False}
show_exact_shap_XR = {"D": True, "W": False, "A": True, "M": False}
show_exact_shap_SR = {"D": True, "W": False, "A": False, "M": False}


show_exact_shap = {
    "LR": show_exact_shap_LR,
    "RR": show_exact_shap_RR,
    "SR": show_exact_shap_SR,
    "XR": show_exact_shap_XR,
}

# --- Datasets vorbereiten
datasets = [
    ("D", df_D, df_D_dict, df_D_dict_colors, exact_shap_values_D, "Diabetes"),
    ("W", df_W, df_W_dict, df_W_dict_colors, exact_shap_values_W, "Wine"),
    ("A", df_A, df_A_dict, df_A_dict_colors, exact_shap_values_A, "Abalone"),
    ("M", df_M, df_M_dict, df_M_dict_colors, exact_shap_values_M, "Music"),
]
bounds_LR = {
    "D": [
    ],
    "W": [
        ("blue",   ":", 19.373605728149414, "Copula SHAP - Lower Bound"),
    ],
    "A": [
        ("green",   ":", 6.15, "Copula SHAP - Lower Bound"),
        ("blue",   ":", 11.269, "Copula SHAP - Lower Bound"),
    ],
    "M": [
        
    ]
}

bounds_RR = {
    "D": [
        ("purple", ":", 8, "Sample SHAP - Lower Bound"),
        ("green",  "-", 7, "Gaussian SHAP - Upper Bound"),
        ("blue",  "-", 7.6, "Copula SHAP - Upper Bound")
    ],
    "W": [
        ("green",  "-", 86, "Gaussian SHAP - Upper Bound"),
        ("purple", ":", 123.42659, "Sample SHAP - Lower Bound"),
        ("orange", ":", 103.550527,"Kernel SHAP - Lower Bound"),
    ],
    "A": [
        ("purple", ":", 72, "Sample SHAP - Lower Bound"),
        ("green",  "-", 105, "Gaussian SHAP - Upper Bound"),
        ("blue",  "-", 205, "Copula SHAP - Upper Bound")],
    "M": [("green",  "-", 15.671, "Gaussian SHAP - Upper Bound"),
          ("blue",  "-", 34.280, "Copula SHAP - Upper Bound")
        ]
}

bounds_SR = {
    "D": [("green",  "-", 34, "Gaussian SHAP - Upper Bound"),
        ("blue",  "-", 37, "Copula SHAP - Upper Bound")],
    "W": [],
    "A": [("orange", ":", 1027.620,"Kernel SHAP - Lower Bound")],
    "M": [
        ("blue",   "-",    60, "Copula SHAP - Upper Bound"),
        ("green",  "-",  70.050, "Gaussian SHAP - Upper Bound"),
    ]
}

bounds_XR = {
    "D": [
        ("blue",   "-",    6.7, "Copula SHAP - Upper Bound"),
        ("green",  "-",  6.449, "Gaussian SHAP - Upper Bound"),
    ],
    "W": [("green",  "-",  70.76, "Gaussian SHAP - Upper Bound"),],
    "A": [],
    "M": [
        ("blue",   "-",    27.39117693901062, "Copula SHAP - Upper Bound"),
        ("green",  "-",  26, "Gaussian SHAP - Upper Bound"),
    ]
}

bounds = {
    "LR": bounds_LR,
    "RR": bounds_RR,
    "SR": bounds_SR,
    "XR": bounds_XR,
}


# ----------------------------------------
# Figure mit 4 Subplots nebeneinander
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes = axes.flatten()

for index, (ax, (data_key, df_ds, ds_dict, data_colors, exact_values, data_name)) in enumerate(zip(axes, datasets)):
    exact_shap_value = None

    # Daten sammeln
    for approximation in approximations:
        df_sel = df_ds[(df_ds["index"].str.contains(f"{approximation}_")) & (df_ds["index"].str.contains(f"_{model}_"))]
        ds_dict[f"{approximation}_{model}"] = df_sel
        data_colors[f"{approximation}_{model}"] = colors[approximation]
        exact_shap_value = exact_values.get(model, None)
        model_name = model_names[model]

    # Scatter + Lineplots
    for key in ds_dict.keys():
        ax.scatter(ds_dict[key]["Running_Time"], ds_dict[key]["Stability_Score"],
                   color=data_colors[key], label=shap_names[key.split('_')[0]])
        _dfp = ds_dict[key].sort_values("Running_Time")
        ax.plot(_dfp["Running_Time"], _dfp["Stability_Score"],
                color=data_colors[key], linestyle='-')

    # Labels & Titel
    if index == 0:
        ax.set_ylabel("Mean 95% Confidence Interval Width (SHAP Values)", fontsize=9)
    ax.set_title(f"{data_name} Dataset")

    # Rote Linie = Exact SHAP (nur wenn erlaubt)
    if show_exact_shap[model][data_key] and exact_shap_value is not None:
        ax.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")

    # Bounds-Linien
    for color_b, ls_b, x_b, lbl_b in bounds[model].get(data_key, []):
        ax.axvline(x=x_b, color=color_b, linestyle=ls_b, linewidth=1, label=lbl_b, alpha=0.6)


# ----------------------------------------
# Gemeinsame Legende
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l

reference_set = set()
uniq_handles, uniq_labels = [], []
for h, l in zip(handles, labels):
    if l not in reference_set:
        uniq_handles.append(h)
        uniq_labels.append(l)
        reference_set.add(l)

fig.legend(uniq_handles, uniq_labels, loc="lower center", ncol=3, fontsize=10)

# Gemeinsame X-Achse und Layout
fig.text(0.5, 0.18, "Running Time in Seconds", ha="center", fontsize=12)
plt.tight_layout(rect=[0.01, 0.18, 0.99, 0.95])
plt.suptitle(f"Stability Comparison (10 runs) - {model_names[model]}", fontsize=18)
plt.show()