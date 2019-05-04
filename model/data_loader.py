import torchvision
import pickle
import torch
import PIL
import os
import cv2

# for image normalization
imagenet_stats = ([0.485, 0.456, 0.406], 
                  [0.229, 0.224, 0.225])

sz = 224 

# list of augmentations
augm = {
    "train": torchvision.transforms.Compose([
       torchvision.transforms.Resize(sz),
       torchvision.transforms.RandomHorizontalFlip(),
       torchvision.transforms.RandomResizedCrop(sz),
       torchvision.transforms.RandomRotation(30),
       torchvision.transforms.RandomGrayscale(),
       torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize(*imagenet_stats),
    ]),
    "valid": torchvision.transforms.Compose([
        torchvision.transforms.Resize(sz),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*imagenet_stats),
    ]),
}


def save_augmentation(d, pkl_path):
    """Saves dict of augmentations in pickle file. We need this to log 
    which augm was used.
    Inputs:
    - d: (dict) of augmentation values
    - pic: (string) path to pickle file."""
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(d, f)


def fetch_dataloader(types, data_dir, params):
    """Fetches the DataLoader object for each type in types from data_dir.
    Inputs:
    - types: (list) has one or more of 'train', 'valid', 'test' depending 
      on which data is required
    - data_dir: (string) directory containing the dataset 
    - params: (Params) hyperparameters
    Returns:
    - data : (dict) contains the DataLoader object for each type in types."""
    
    # helper lambda function to open image
#     open_image = lambda x: PIL.Image.open(x)
    open_image = lambda x: PIL.Image.fromarray(cv2.imread(x))  
    dataloaders = {}
    for split in ['train', 'valid', 'test']:
        if split in types:
            path = os.path.join(data_dir, split)
                        
            # use the train_transformer if training data, 
            # else use eval_transformer without random flip
            if split == 'train':
                dl = torch.utils.data.DataLoader(
                    torchvision.datasets.ImageFolder(path, augm["train"], loader=open_image),
                    batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, 
                    pin_memory=params.cuda, drop_last=True)
            else:
                dl = torch.utils.data.DataLoader(
                    torchvision.datasets.ImageFolder(path, augm["valid"], loader=open_image),
                    batch_size=params.batch_size if split=="valid" else params.bs_test , 
                    shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda, 
                    drop_last=True)
            dataloaders[split] = dl
    
    # test mode to check that model overfit the data
#     dataloaders['train'].dataset.samples = dataloaders['train'].dataset.samples[:2500]
    return dataloaders