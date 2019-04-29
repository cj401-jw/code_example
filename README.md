# Structure of the code
This is a template of project code leveraged from [Stanford cs230 Deep Learning](https://cs230-stanford.github.io/project-code-examples.html) course and updated for my specific purposes.

```
├── data                    <- folder could contain different datasets
│   └── imagenet
│       ├── train 
│       ├── valid
│       └── test
├── notebooks               <- if you need to debug or try new idea, poc
├── experiments             
│   └── experiment_name     <- each experiment has it own folder
│       └── params.json     <- file contain all params for training
├── model 
│   ├── net.py              <- model, layers to build a model and all related to model magic
│   ├── metric.py           <- all metrics defined here and put all together in metrics dict
│   └── data_loader.py      <- augm storied here as well as all stuff related to dataset
├── include.py              <- script contains settings, imported modules, imagenet stats for norm and so on 
├── synthesize_results.py   <- plot results in tabular for several trained models or after hyperparams search
├── plot_results.py         <- plot results in graphical way for several trained models or after hyperparams search
├── search_hyperparams.py   <- run train.py multiple times with different hyperparameters
├── evaluate.py             <- run model to evaulate on test set   
├── train.py                <- main script which runs all magic, loss_func and optimizer defined here
└── utils.py                <- supporter funcs
```

## Dataset
Dataset storied in `data/` folder. In case if use have several datasets then you should have folder for each dataset with following folders: train, valid, test (optional). Here is a structure of the data:
```
Data/
  Imagenet/
    train/
      apple/
        image.jpg
      banana/
      ...
    valid/
      apple/
      banana/
      ...
    test/
      ...
```
Note: The scructure of the data could be different in that case you will need to update `fetch_dataloader()` function in `data_loader.py`.

## Experiment parameters
TODO: While experiments it's obvious that net.py script should be in experiment folder. When you performed some experiment and then changed model (some tiny chenges) you can forget what model was. To fix it it's better to have a model folder within each experiment you want to log a model.


To compare experiment results we need to log all parameters which was used while training. We do this through `params.json` file. Each experiment has it's own folder and `params.json` under it. Here is detailed description of this file:
```json
{
    "arch": "resnet18",
    "pretrain": 1,
    "n_clas": 9,
    "im_sz": 224,
    "batch_size": 128,
    "bs_test": 16,
    "learning_rate": [     learning rate for each layer groups
        0.002,
        0.002,
        0.002
    ],
    "wd": 0.0001,
    "ps": 0.5,             dropout for head classifier layer
    "num_epochs": 10,
    "num_workers": 4,
    "cuda": 1,
    "scheduler": "one_cycle",
    "div_factor": 25,
    "pct_start": 0.3,
    "save_summary_steps": 1,
    "early_stopping": {
        "patience": 10,
        "monitor_metric": "loss",
        "min_delta": 0.01,
        "minimize": 1
    },
    "hyper_search": {     we use it when run search_hyperparams.py
        "ps": [
            0.5,
            0.75
        ]
    }
}
```
This json file we loading in `train.py` script and propogate throught all methods which require these params. If you need to clarify where these parameters go from this json you should directly go to `train.py`.

## __Running experiments__

### __Train single model__</br>
Create a folder with experiment name under `experiments` folder with `params.json`. It should looks like `experiments/resnet18/params.json`. Then just run the experiment:
```
python train.py --data_dir data/imagenet/ --model_dir experiments/resnet18/
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate metrics (defined in `model/metirc.py`) on the validation set. While training you can monitor training through real time ploting which automaticly will arise after start training. </br> 
The structure of completed experiment might look like this (try to give meaningful names to the directories depending on what experiment you are running):
```
├── experiments             
│   └── resnet18     
│       ├── metrics_val_last_weights.json        <- weights saved from the 5 last epochs
│       ├── metrics_val_best_weights.json        <- best weights (based on valid accuracy)
│       ├── params.json             <- the list of hyperparameters, in json format
│       ├── history.json            <- history of training 
│       ├── augm.pkl                <- used augmentation
│       ├── train.log               <- he training log (everything we print to the console)
│       ├── train_history.png       <- train summaries in figure
```

### __Hyper parameters search__
This script will run `train.py` several times with different hyperparams. Pay attention that hyperparams to search should be defined in `params.json` like in the following example: 
```json
{
    "arch": "resnet18",
    ...
    "hyper_search": {     
        "learning_rate": [
            1e-4,
            1e-3,
        ]
    }
}
```
Also folder for experiment should be created. Folder name should be equal to searning parameter for example `experiments/resnet18/learning_rate/` , put defined above json file in this folder and run:
```
python search_hyperparams.py --data_dir data/imagenet/ --parent_dir experiments/resnet18/learning_rate/
```
It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`. </br>
The structure of completed params searching might look like this:
```
├── experiments             
│   └── resnet18     
│       └── learning_rate
│           ├── params.json               <- params used to init model, each folder has own json with approp search values
│           ├── history_plots.png         <- summary plot with all experiments  
│           ├── results.md                <- table of metrics for all values 
│           ├── learning_rate_0.001       <- folder and folder beneath has the same structure as after run train.py
│           ├── learning_rate_0.0001      
│           └── learning_rate_0.00001     
```
So as you can see it automaticaly creates folder for each value of search variable (parameter). The scrtucture 
of these folders are the same as after running `train.py`.

__Important notes__: 
1. You can run for example several different architectures not just params. 
2. Pay attention we can do search only for those params which we have in this `params.json` file. 
3. One run one variable to search. To run search for several variable is in todo list.

## __Debugging__
To check that new architecture working we can overfit it on a small dataset. To dicrease a dataset uncomment line of code in the `model/data_loader.py`. Also turn off or set lighter augmentation. As result model should show 100 % accuracy or any metric you use.

### __Evaluation on the test set__ 
Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```

### __Display the results__ 
of the hyperparameters search in a nice format. This script runs automaticaly within searching hyperparasm but also could be runned separately.
```
python synthesize_results.py --parent_dir experiments/learning_rate
```
It will create search params results in tabular format. </br> 
If you want to see results in __graphical__ way please run.
```
python plot_results.py --parent_dir experiments/learning_rate
```
This script also runs automaticaly within searching hyperparams.



__TODO__:
- describe how to use this code                  <- DONE
- clean up code in scripts                       
- update Early Stopping
- compare performance with fast.ai library
- run search_params.py with weights
- add monitoring of activations/gradients
- add monitoring of weights/updates magnitude (histogram of all layers)
