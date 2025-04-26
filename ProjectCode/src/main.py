import sys
from pathlib import Path
import argparse

src_path = str(Path(__file__).parent/"src")
sys.path.append(src_path)

# import training function
from train import train_models

#impoort shap functions
from run_shap import run_and_save_shap_values
from shap_approximations.shap_loader import get_shap_values

#import evaluation functions
from evaluations.stability import compute_stability_score
from evaluations.result_organizer import print_restults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["T", "R", "E", "L", "P"], type=str, required=True, help="T = Training all the models on all data sets. R = Run one model on one Shap Approximation and saving runtime. E = Evaluate one SHAP approximation on one Method. L = Load SHAP values. P = Print all Evaluation Results")
    parser.add_argument("--data", choices=["A", "W"], type=str, help="A = Abalone, W = Wine Quality")
    parser.add_argument("--model", choices=["LR", "LO", "RR", "RC", "SR", "SC", "XR", "XC"], type=str, help="LR = LinRegression, LO = LogRegression, RR = RFRegressor, RC = RFClassifier, SR = SVMRegressor, SC = SVMClassifier, XR = XGBRegressor, XC = XGBClassifier")
    parser.add_argument("--approximation", choices=["K", "S", "B", "C", "G", "E"], type=str, help="K = Kernel SHAP, S = Sampling SHAP, B = Bounded Sampling, C = Copula SHAP, G = Gauss Copula SHAP, E = Exact SHAP")
    parser.add_argument("--nsamples", type=int, help="Number of coalitions for the SHAP approximation")
    parser.add_argument("--number_of_runs", type=int, help="Number of individual SHAP calculations for the same model, data and approximation for stability evaluation")
    parser.add_argument("--eval_method", choices=["F", "R", "S"], type=str, help="F = Fidelity, R = RBO, S = Stability")
    
    args = parser.parse_args()
    
    match args.mode:
        case "T":
            train_models()
        case "R":
            if not all([args.data, args.model, args.approximation, args.nsamples]):
                print("Not all arguments provided for running an approximation.")
            else:
                if not (args.number_of_runs):
                    run_and_save_shap_values(args.data, args.model, args.approximation, args.nsamples)
                else:
                    run_and_save_shap_values(args.data, args.model, args.approximation, args.nsamples, args.number_of_runs)
                return
            print("R")
        case "E":
            if not (args.eval_method):
                print("Evaluation Method needs to be defined.")
            else:
                match args.eval_method:
                    case "F":
                        print("Fidelity")
                    case "R":
                        print("RBO")
                    case "S":
                        if all([args.data, args.model, args.approximation, args.nsamples, args.number_of_runs]):
                            print(compute_stability_score(args.data, args.model, args.approximation, args.nsamples, args.number_of_runs))
                        else:
                            print("Not all arguments provided for running an Stability evaluations are provided.")
                    case _:
                        print("Invalid Evaluation Method.")
        case "L":
            if not all([args.data, args.model, args.approximation, args.nsamples]):
                print("Not all arguments provided for running an approximation.")
            else:
                if not (args.number_of_runs):
                    print(get_shap_values(args.data, args.model, args.approximation, args.nsamples))
                else:
                    for run_id in range(args.number_of_runs):
                        print(get_shap_values(args.data, args.model, args.approximation, args.nsamples, run_id+1))
        case "P":
            print_restults()

if __name__ == "__main__":
    main()
    

