"""Class and supporter function to plot results of training."""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments',
                    help='Directory containing results of experiments')


class Plotter():
    """Class helps to plot results from different experiments.
    Inputs:
    - parent_dir: (string) path to directory containing experiments results."""
    
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.dicts = self.read_history_dicts()
        # list of experiments which have history json!
        self.experiments = list(self.dicts.keys()) 
        
    def read_history_dicts(self):
        """Parse and append to gloabl dict history files from all available 
        experiments. Returns dict where keys is experiment names.
        Inputs:
        - dicts: (dict) containing parsed history.json from all experiments in 
          parent_dir.
        Returns:
        - dicts: (dict) containing metrics value for train and validation."""
        
        dicts = {}
        # list of folders of all experiments, e.g. even empty
        experiments_folder = [(f'{self.parent_dir}/{exp}') for exp in next(os.walk(self.parent_dir))[1]]
        
        # walk through each experiment 
        for experiment_folder in experiments_folder:

            # parse experiment name and parse files from exp folder
            exp_name = experiment_folder.split('/')[-1]
            files_in_folder = os.listdir(experiment_folder)

            # parse and append history file to glob dict
            if 'history.json' in files_in_folder:
                with open(f'{experiment_folder}/history.json', 'r') as f:
                    history = json.load(f)
                dicts[exp_name] = history    
        return dicts
    
    def realtime_plot_history_dicts(self):
        pass

    def plot_history_dicts(self):
        """Takes history from all experiments and save figure with them
        on the disk.
        Inputs:
        - dicts: (dict) output from read_history_dicts func."""

        # always two cols since one for train second for valid
        metrics = list(self.dicts[list(self.dicts.keys())[0]]['train'].keys())
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 2, figsize=(16, 12))

        # expand metrics for loop ['acc', 'loss'] -> ['acc', 'acc', 'loss', 'loss' ]
        metrics_double = np.ravel( list(zip(metrics, metrics)) )  
        assert len(metrics_double) == len(metrics*2), 'Metrics len should be metrics*2'
        
        # num metrics equal to num rows
        for i, (ax, metric) in enumerate(zip(axes.flatten(order="C"), metrics_double)):

            state = 'train' if i % 2 == 0 else 'valid'
            ax.set_title(f'{state}_{metric}')
            ax.set_xlabel('epoch'); ax.grid()
            
            # each plot plot all experiments
            for experiment in self.experiments:
                ax.plot(self.dicts[experiment][state][metric], label=f'{experiment}')
            ax.legend()

        # save figure on the dist
        figure_file_name = f'{self.parent_dir}history_plots.png'
        plt.savefig(figure_file_name) 
        print(f'Figure save to {figure_file_name}')
        

if __name__ == "__main__":
    args = parser.parse_args()
    plotter = Plotter(args.parent_dir)
    plotter.plot_history_dicts()

          
          
          
          
          