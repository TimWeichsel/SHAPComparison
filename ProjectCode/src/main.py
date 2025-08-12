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
from evaluations.rbo import compute_rbo_score
from evaluations.fidelity import compute_fidelity_score
from evaluations.result_organizer import print_results
from evaluations.result_organizer import get_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["T", "R", "E", "L", "P", "Z"], type=str, required=True, help="T = Training all the models on all data sets. R = Run one model on one Shap Approximation and saving runtime. E = Evaluate one SHAP approximation on one Method. L = Load SHAP values. P = Print all Evaluation Results")
    parser.add_argument("--data", choices=["A", "D", "M", "W"], type=str, help="A = Abalone (Regression), D = Diabetes (Regression), M = Music Origin (Regression), W = Wine Quality (Classification & Regression)")
    parser.add_argument("--model", choices=["LR","LO", "RR", "RC", "SR", "SLR", "SC", "XR", "XC"], type=str, help="LR = LinRegression, LO = LogRegression, RR = RFRegressor, RC = RFClassifier, SR = SVMRegressor with rbf Kernel, SLR = SVMRegressor with linear Kernel, SC = SVMClassifier, XR = XGBRegressor, XC = XGBClassifier")
    parser.add_argument("--approximation", choices=["K", "S", "C", "G", "E"], type=str, help="K = Kernel SHAP, S = Sampling SHAP, C = Copula SHAP, G = Gauss Copula SHAP, E = Exact SHAP")
    parser.add_argument("--nsamples", type=int, help="Number of coalitions for the SHAP approximation")
    parser.add_argument("--number_of_runs", type=int, help="Number of individual SHAP calculations for the same model, data and approximation for stability evaluation")
    parser.add_argument("--eval_method", choices=["F", "R", "S"], type=str, help="F = Fidelity, R = RBO, S = Stability")
    
    args = parser.parse_args()
    
    match args.mode:
        case "Z":
            from run_comparison.run_script import run_approximations
            from run_comparison.run_script import run_evaluations
            from run_comparison.run_script import run_stability
            run_approximations()
            run_evaluations()
            run_stability()
            
            

        case "T":
            train_models()
        case "R":
            if args.approximation == "E":
                if not all([args.data, args.model]):
                    raise ValueError("Not all arguments provided for running an approximation.")
                else:
                    run_and_save_shap_values(args.data, args.model, args.approximation, None)
            else:
                if not all([args.data, args.model, args.approximation, args.nsamples]):
                    raise ValueError("Not all arguments provided for running an approximation.")
                else:
                    if not (args.number_of_runs):
                        run_and_save_shap_values(args.data, args.model, args.approximation, args.nsamples)
                    else:
                        run_and_save_shap_values(args.data, args.model, args.approximation, args.nsamples, args.number_of_runs)
                    return
        case "E":
            if not (args.eval_method):
                raise ValueError("Evaluation Method needs to be defined.")
            else:
                match args.eval_method:
                    case "F":
                        print(compute_fidelity_score(args.data, args.model, args.approximation, args.nsamples))
                    case "R":
                        print(compute_rbo_score(args.data, args.model, args.approximation, args.nsamples))
                    case "S":
                        if all([args.data, args.model, args.approximation, args.nsamples, args.number_of_runs]):
                            print(compute_stability_score(args.data, args.model, args.approximation, args.nsamples, args.number_of_runs))
                        else:
                            raise ValueError("Not all arguments provided for running an Stability evaluations are provided.")
                    case _:
                        raise ValueError("Invalid Evaluation Method.")
        case "L":
            if not all([args.data, args.model, args.approximation]):
                raise ValueError("Not all arguments provided for running an approximation.")
            else:
                if not (args.number_of_runs):
                    shap_values = get_shap_values(args.data, args.model, args.approximation, args.nsamples)
                else:
                    for run_id in range(args.number_of_runs):
                        print(get_shap_values(args.data, args.model, args.approximation, args.nsamples, run_id+1)) 
                print(type(shap_values))
                print(shap_values.shape)
                print(shap_values[0:10,:])
        case "P":
            eval_method = None
            approximation_method = ""
            model_name = ""
            if (args.eval_method):
                eval_method = args.eval_method
            if (args.approximation):
                approximation_method = args.approximation
            if (args.model):
                model_name = args.model
            if not (args.data):
                print_results(eval_method=eval_method, approximation_name=approximation_method, model_name=model_name)
            else:
                print_results(args.data, eval_method=eval_method, approximation_name=approximation_method, model_name=model_name)

if __name__ == "__main__":
    main()
    

