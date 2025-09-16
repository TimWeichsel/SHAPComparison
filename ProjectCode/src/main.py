import sys
from pathlib import Path
import argparse

src_path = Path(__file__).parent #get src path
sys.path.insert(0, str(src_path)) #prioritize src path

shapr_path = src_path / "external" / "shapr" / "python" #path to python shapr directory
sys.path.insert(1, str(shapr_path)) #add python shapr directory

# import training function
from train import train_models


#impoort shap functions
from run_shap import run_and_save_shap_values
from shap_approximations.shap_loader import get_shap_values

#import evaluation functions
from evaluations.stability import compute_stability_score
from evaluations.rbo import compute_rbo_score
from evaluations.fidelity import compute_fidelity_score_MSE
from evaluations.fidelity import compute_fidelity_score_MAE
from evaluations.result_organizer import print_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["Z","T", "C", "E", "L", "P"], type=str, required=True, help="T = Training all the models on all data sets. C = Compute SHAP (Approximation) values for one model and saving its total runtime. E = Evaluate one SHAP approximation on one Method (default: Runtime). L = Load SHAP values. P = Print all Results of one Evaluation Metric (default: Runtime)")
    parser.add_argument("--data", choices=["A", "D", "M", "W"], type=str, help="A = Abalone (Regression), D = Diabetes (Regression), M = Music Origin (Regression), W = Wine Quality (Classification & Regression)")
    parser.add_argument("--model", choices=["LR", "RR", "SR", "XR"], type=str, help="LR = LinRegression, RR = RFRegressor, SR = SVMRegressor, XR = XGBRegressor")
    parser.add_argument("--approximation", choices=["K", "S", "C", "G", "E", "C100", "G100"], type=str, help="K = Kernel SHAP, S = Sampling SHAP, C = Copula SHAP (MC_Samples = 10), G = Gauss Copula SHAP (MC_Samples = 10), C100 = Copula SHAP (MC_Samples = 1000), G100 = Gauss Copula SHAP (MC_Samples = 1000), E = Exact SHAP")
    parser.add_argument("--nsamples", type=int, help="Number of coalitions for the SHAP approximation")
    parser.add_argument("--number_of_runs", type=int, help="Number of individual SHAP calculations for the same model, data and approximation for stability evaluation")
    parser.add_argument("--eval_method", choices=["F", "F_MAE", "R", "S"], type=str, help="F = Fidelity (Mean Squared Error), F_MAE = Fidelity (Mean Absolut Error) R = RBO, S = Stability")
    parser.add_argument("--instance", type=int, help="Index for listing only one SHAP instance")
    
    args = parser.parse_args()
    
    match args.mode:
        # Training all the models on all data sets
        case "T":
            train_models()
        
        # Compute SHAP (Approximation) values for one model and saving its total runtime.
        case "C":
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
                        run_and_save_shap_values(args.data, args.model, args.approximation, args.nsamples) # Single Run
                    else:
                        run_and_save_shap_values(args.data, args.model, args.approximation, args.nsamples, args.number_of_runs) # Multiple Runs for Stability
                    return
                
        # Evaluate one SHAP approximation on one Method (default: Runtime)
        case "E":
            if not (args.eval_method):
                raise ValueError("Evaluation Method needs to be defined.")
            else:
                match args.eval_method:
                    case "F":
                        print(compute_fidelity_score_MSE(args.data, args.model, args.approximation, args.nsamples))
                    case "F_MAE":
                        print(compute_fidelity_score_MAE(args.data, args.model, args.approximation, args.nsamples))
                    case "R":
                        print(compute_rbo_score(args.data, args.model, args.approximation, args.nsamples))
                    case "S":
                        if all([args.data, args.model, args.approximation, args.nsamples, args.number_of_runs]):
                            print(compute_stability_score(args.data, args.model, args.approximation, args.nsamples, args.number_of_runs))
                        else:
                            raise ValueError("Not all arguments provided for running an Stability evaluations are provided.")
                    case _:
                        raise ValueError("Invalid Evaluation Method.")
        
        # Load SHAP values
        case "L":
            if not all([args.data, args.model, args.approximation]):
                raise ValueError("Not all arguments provided for running an approximation.")
            else:
                if not (args.nsamples) and args.approximation!="E":
                    raise ValueError("Number of samples needs to be verified, if not Exact SHAP is used.")
                if not (args.number_of_runs):
                    if args.nsamples:
                        shap_values = get_shap_values(args.data, args.model, args.approximation, nsamples=args.nsamples, instance_index=args.instance)
                        print(shap_values)
                    else:
                        shap_values = get_shap_values(args.data, args.model, args.approximation, instance_index=args.instance)
                        print(shap_values)
                else:
                    for run_id in range(args.number_of_runs):
                        print(get_shap_values(args.data, args.model, args.approximation, nsamples=args.nsamples, run_id=run_id+1, instance_index=args.instance)) 
        
        # Print all Results of one Evaluation Metric (default: Runtime)
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
    

