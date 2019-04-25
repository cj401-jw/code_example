# Template of project code
This is a template of project code leveraged from Stanford cs230 course and updated for my specific purposes.

TODO:
- describe how to use this code
- update Early Stopping
- compare performance with fast.ai library
- run search_params.py with weights
- add monitoring of activations/gradients
- add monitoring of weights/updates magnitude (histogram of all layers)

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
To compare experiment results we need to log all parameters which was used while training. We do this through `params.json` file. Each experiment has it's won folder `params.json` under it. Here is detailed description of this file:
``json
{
    "arch": "resnet18",
    "pretrain": 1,
    "n_clas": 9,
    "im_sz": 224,
    "batch_size": 128,
    "bs_test": 16,
    "learning_rate": [     # learning rate for each layer groups
        0.002,
        0.002,
        0.002
    ],
    "wd": 0.0001,
    "ps": 0.5,             # dropout for head classifier layer
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
    "hyper_search": {     # we use it when run search_hyperparams.py
        "ps": [
            0.5,
            0.75
        ]
    }
}
```


## Quickstart

1. __Create a folder for experiment and parameters__: Create a folder with experiment name under `experiments` folder with `params.json`. Put template structure for json file. 
2. __Run experiment__: To run the experiment just 
