"""Evaluates the model"""

from include import *
import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
from model import data_loader, net, metric

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Inputs:
    - metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batchl
    - params: (Params) hyperparametersl
    - num_steps: (int) number of batches to train on, each of size params.batch_size."""

    # set model to evaluation mode
    model.eval()
    torch.set_grad_enabled(False)
    
    torch.cuda.empty_cache() 

    # summary for current eval loop
    summ = []
    
    # compute metrics over the dataset
    for images, labels in tqdm.tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # compute model output
        scores = model(images)
        loss = loss_fn(scores, labels)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric:metrics[metric](scores, labels) for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
        
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model and optimizer
    if params.arch == "resnet18": model = net.resnet18(num_classes=params.n_clas, pretrained=params.pretrain).to(device)             
    elif params.arch == "se_resnet18": model = net.se_resnet18(num_classes=params.n_clas, pretrained=params.pretrain).to(device)
    elif params.arch == "cbam_rn18": model = net.cbam_resnet18(num_classes=params.n_clas, pretrained=params.pretrain).to(device)
    elif params.arch == "resnet152": model = net.resnet152(num_classes=params.n_clas, pretrained=params.pretrain).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = metric.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)