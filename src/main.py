import sys
from pathlib import Path
import argparse

#set paths to start the program from main.py
src_path = Path(__file__).parent #get src path
sys.path.insert(0, str(src_path)) #prioritize src path
shapr_path = src_path / "external" / "shapr" / "python" #path to python shapr directory
sys.path.insert(1, str(shapr_path)) #add python shapr directory

#import training function
from train import train_models

#import shap functions
from run_shap import run_and_save_shap_values
from shap_approximations.shap_loader import get_shap_values

#import evaluation functions
from evaluations.stability import compute_stability_score
from evaluations.rbo import compute_rbo_score
from evaluations.fidelity import compute_fidelity_score_MSE
from evaluations.fidelity import compute_fidelity_score_MAE
from evaluations.result_organizer import print_results

def main():
    parser = argparse.ArgumentParser(description='''
    A Comparative Evaluation of Model-Agnostic SHAP Approximations
    --------------------------------------------------------------
    Bachelor Thesis Project by BSc. Tim Weichselberger (University of Vienna)
    supervised by Prof. Ass.-Prof. Dipl.-Ing. Dr.techn. Sebastian Tschiatschek
                                    
    - Description:
    This thesis' project investigates how well standard model-agnostic SHAP approximations align with the evaluation properties of Degree of Importance (RBO), Fidelity(MAE) and Stability(Confidence Interval) across different machine learning models.
                                        
    - Usage of the script:
    Run the main.py file with "--help" to see all the necessary information.
    The script can be use to train models, compute SHAP values, evaluate SHAP values, list SHAP values and print results.
    To run the script, the libraries specified in requirements.txt need to be installed as well as R with the shapr package (see README.md for detailed information).
    Different Modes can be selected via the --mode argument to perform the different tasks.
    For each mode, different required arguments are needed and some optional arguments can be specified.
    More information can be found below.
    ''',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", choices=["T", "C", "E", "L", "P"], type=str, required=True, help='''
    - T: Training all the models on all data sets. (No additional arguments needed)
                        
    - C: Compute SHAP (Approximation) values for one model and saving its total runtime
    -> Required: --data, --model, --approximation, --nsamples (except for Exact SHAP)
    -> Optional: --number_of_runs
                            
    - E: Evaluate one SHAP approximation on one Method (default: Runtime)
    -> Required: --eval_method, --data, --model, --approximation (Exact SHAP not allowed) 
    -> Optional: --number_of_runs
                            
    - L: List SHAP values (optional specifying a particular instance)
    -> Required: --data, --model, --approximation, --nsamples (except for Exact SHAP) 
    -> Optional: --instance, --number_of_runs
                            
    - P: Print all Results of one Evaluation Metric (default: Runtime) and filter optional 
    -> Optional arguments: --eval_method, --data, --model, --approximation
                            
                        ''')
    parser.add_argument("--data", choices=["A", "D", "M", "W"], type=str, help='''
    - A: Abalone (Regression),
    - D: Diabetes (Regression),
    - M: Music Origin (Regression),
    - W: Wine Quality (Classification & Regression)
                        
                        ''')
    parser.add_argument("--model", choices=["LR", "RR", "SR", "XR"], type=str, help='''
    - LR: Linear Regression 
    - RR: Random Forest Regressor 
    - SR: Support Vector Machine Regressor
    - XR: XGBoost Regressor
                        
                        ''')
    parser.add_argument("--approximation", choices=["K", "S", "C", "G", "E", "C100", "G100"], type=str, help='''
    - K: Kernel SHAP
    - S: Sampling SHAP
    - C: Copula SHAP (MC_Samples = 10)
    - G: Gaussian Copula SHAP (MC_Samples = 10)
    - C100: Copula SHAP (MC_Samples = 1000)
    - G100: Gaussian Copula SHAP (MC_Samples = 1000)
    - E: Exact SHAP
                        
                        ''')
    parser.add_argument("--nsamples", type=int, help='''
    Number of coalitions/samples used for the SHAP approximation''')
    parser.add_argument("--number_of_runs", type=int, help='''
    Number of individual SHAP calculations for the same model, data, nsamples and approximation to evaluate Stability''')
    parser.add_argument("--eval_method", choices=["F", "F_MAE", "R", "S"], type=str, help='''
    - F: Fidelity (Mean Squared Error)
    - F_MAE: Fidelity (Mean Absolute Error)
    - R: RBO
    - S: Stability
                        
                        ''')
    parser.add_argument("--instance", type=int, help='''
    Index for listing only one SHAP instance''')
    
    args = parser.parse_args()
    
    match args.mode:
        # Training all the models on all data sets
        case "T":
            train_models()
        
        # Compute SHAP (Approximation) values for one model and saving its total runtime.
        case "C":
            if args.approximation == "E":
                if not all([args.data, args.model]):
                    raise ValueError("Mode Compute SHAP: Not all arguments provided for running an approximation.")
                else:
                    run_and_save_shap_values(args.data, args.model, args.approximation, None)
            else:
                if not all([args.data, args.model, args.approximation, args.nsamples]):
                    raise ValueError("Mode Compute SHAP: Not all arguments provided for running an approximation.")
                else:
                    if args.number_of_runs is None:
                        run_and_save_shap_values(args.data, args.model, args.approximation, args.nsamples) # Single Run
                    else:
                        run_and_save_shap_values(args.data, args.model, args.approximation, args.nsamples, args.number_of_runs) # Multiple Runs for Stability
                
        # Evaluate one SHAP approximation on one Method (default: Runtime)
        case "E":
            if not (args.eval_method):
                raise ValueError("Mode Evaluate SHAP: Evaluation Method needs to be defined.")
            if not all([args.data, args.model, args.approximation, args.nsamples]):
                raise ValueError("Mode Evaluate SHAP: Not all arguments provided for running an approximation.")
            if args.approximation == "E":
                raise ValueError("Mode Evaluate SHAP: Exact SHAP cannot be evaluated.")
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
                            raise ValueError("Mode Evaluate SHAP: Not all arguments provided for running an Stability evaluations are provided.")
                    case _:
                        raise ValueError("Mode Evaluate SHAP: Invalid Evaluation Method.")
        
        # List SHAP values
        case "L":
            if not all([args.data, args.model, args.approximation]):
                raise ValueError("Mode List SHAP: Not all arguments provided for running an approximation.")
            else:
                if args.nsamples is None and args.approximation!="E":
                    raise ValueError("Mode List SHAP: Number of samples needs to be verified, unless Exact SHAP is used.")
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
    

