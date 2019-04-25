# Template of project code
This is a template of project code leveraged from Stanford cs230 course and updated for my specific purposes.

TODO:
- describe how to use this code
- update Early Stopping
- compare performance with fast.ai library
- run search_params.py with weights
- add monitoring of activations/gradients
- add monitoring of weights/updates magnitude (histogram of all layers)

## Folder structure with description.
TODO

This project template could be applied for any computer vision task with tiny changes.

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

## Logistics
1. __metrics__: All metrics should be defined in `model/metric.py`
2. __loss function__: defined in `train.py`. If it custom than could be defined in `net.py`
3. __augmentation__: defined in `model/data_loader.py`. 
4. __optimizer__: defined in `model/train.py`. 
5. __architecture__: model, layers to build a model and all related to model magic are in `model/net.py`. 


## Quickstart

__Train single model.__
1. Create a folder with experiment name under `experiments` folder with `params.json`. 
2. To run the experiment just 
```
python train.py --data_dir data/imagenet/ --model_dir experiments/resnet18/
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the validation set. While training you can monitor training through real time ploting which automaticly will arise after start training. __TODO__ put description of all files within experiment folder after finish training.

__Hyper parameters search.__
To run hyper params search run: 
```
python search_hyperparams.py --data_dir data/imagenet/ --parent_dir experiments/resnet18/learning_rate/
```
It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.


