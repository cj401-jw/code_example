# Template of project code
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
├── search_hyperparams.py   
├── evaluate.py             <- run model to evaulate on test set   
├── train.py                <- main script which runs all magic, loss_func and optimizer defined here
└── utils.py                <- supporter funcs
```

## Dataset
Dataset storied in Data folder. In case if use have several datasets then you should have folder for each dataset with following folders: train, valid, test (optional). Here is a structure of the data:
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
Note: The scructure of the data could be different in that case you will need to update fetch_dataloader function in data_loader.py.

## Experiment parameters
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

## __Train single model__
1. Create a folder with experiment name under `experiments` folder with `params.json`. 
2. To run the experiment just 
```
python train.py --data_dir data/imagenet/ --model_dir experiments/resnet18/
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the validation set. While training you can monitor training through real time ploting which automaticly will arise after start training. After finish training you experiment folder should be:


## __Hyper parameters search__
To run hyper params search run: 
```
python search_hyperparams.py --data_dir data/imagenet/ --parent_dir experiments/resnet18/learning_rate/
```
It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.

## __Compare different architectures__
Case when we comparing different architectures or model parameters.


##__Display the results__ of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```
It will create search params results in tabular format and if you want to see results in graphic way please use
```
python plot_results.py --parent_dir experiments/learning_rate
```

## __Evaluation on the test set__ 
Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```

TODO:
- describe how to use this code
- update Early Stopping
- compare performance with fast.ai library
- run search_params.py with weights
- add monitoring of activations/gradients
- add monitoring of weights/updates magnitude (histogram of all layers)





