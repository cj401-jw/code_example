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

## Experiments parameters
To compare experiment results with need to log all parameters which was used while training. We do this through `params.json` file

## Quickstart

1. __Create a folder for experiment and parameters__: Create a folder with experiment name under `experiments` folder with `params.json`. Put template structure for json file. 
2. __Run experiment__: To run the experiment just 
