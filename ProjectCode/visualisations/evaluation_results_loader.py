import sys
import os
from pathlib import Path

project_start_point = Path(__file__).resolve().parent.parent
src_path = project_start_point / "src"
sys.path.append(str(src_path))

from evaluations.result_organizer import get_results
os.chdir(project_start_point / "src")

def get_evaluation (approximations: list = None, eval_method: list = None):
    df = get_results()
    if eval_method is not None:
        df = df[eval_method]
        
    if approximations is not None:
            return df.loc[approximations]
    return df
    
    
