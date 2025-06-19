from evaluation_results_loader import get_evaluation

df = get_evaluation(eval_method=["Stability_Score","Running_Time"])
print(df)