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

## Quickstart

### 1. Run single experiment.
