"""Peform hyperparemeters search. Folder with experiment should 
contain folder for variable. Will call train.py with different 
values of hyper opt.

Note: In that way we can do hyperparams search of all vars which 
contain params.json. So if we want to do search for some vars we 
need to include it in parasm.json.

TODO: if we use some list as inputs than we have a problem with creating folders.
"""

import argparse
import os
from subprocess import check_call
import sys
import plot_results
import utils
import synthesize_results
from shutil import copyfile
from pathlib import Path

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/resnet18/learning_rate', help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/imagenet/', help="Directory containing the dataset")


def make_source_copy(parent_dir, job_name):
    """We need to take source *.py file from /parent_dir and copy it to 
    the /parent_dir/job_name folder since train.py will search it. Need
    to refactor this code."""
    
    path = Path(parent_dir)
    source_path = next(path.glob("*.py"))
    f_name = source_path.name
    src = str(source_path)
    dst = source_path.parent.joinpath(job_name).joinpath(f_name)
    
    # copy file
    copyfile(src=src, dst=dst)
    

def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Inputs:
    - parent_dir: (string) folder of variable name which we tune.
      experiments/resnet18/learning_rate
    - job_name: value of tuning variable which we testing. 
    - model_dir: (string) directory containing config, weights and log
    - data_dir: (string) directory containing the dataset
    - params: (dict) containing hyperparameters."""
    
    # create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)

    # write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)  
    
    # copy source file to job_name folder
    make_source_copy(parent_dir, job_name)

    # launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir}".format(
        python=PYTHON, model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)  # important method - call variable cmd in shell.


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)  # parent params.json

    search_param_name = list(params.hyper_search.keys())[0]            # get params to search                   
    values_of_search_param = params.hyper_search[ search_param_name ]  # get values to search
    
    for value in values_of_search_param:
        # modify the relevant parameter in params
        params.__dict__[f'{search_param_name}'] = value
        
        # launch job (name has to be unique)
        if type(value) == list: value = '_'.join([str(i) for i in value])  # hand if value is a list
        job_name = "{name}_{value}".format(name=search_param_name, value=value)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)
   
    # save figure with results to the dist
    plotter = plot_results.Plotter(args.parent_dir)
    plotter.plot_history_dicts()
    
    # show results in tabular format
    metrics = dict()
    synthesize_results.aggregate_metrics(args.parent_dir, metrics)
    table = synthesize_results.metrics_to_table(metrics)
    print(table)
    
    # save results in parent_dir/results.md
    save_file = os.path.join(args.parent_dir, "results.md")
    with open(save_file, 'w') as f:
        f.write(table)
