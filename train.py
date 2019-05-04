from include import *
import utils
from model import data_loader, net, metric
from evaluate import evaluate
from pathlib import Path

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/imagenet/', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training") # 'best' or 'train'


def prepare_source_file(model_dir):
    """Load source *.py file with architecture code which located
    in /model_dir. This folder should contain only one *.py file.
    Inputs:
    - model_dir: (string) we use this to find source *.py file
      with architecture.
    Returns:
    - source: (module) python module."""
    
    # find source for architecture in model_dir, should be single *.py file
    source = glob.glob(model_dir+"/*.py")
    assert len(source)==1, f"{source}"
    
    # format string for importing as module
    source = source[0].replace(".py", "").replace("/", "."); 
    assert "/" not in source

    source = __import__(f"{source}", fromlist=[""]); 
    assert params.arch in source.__all__ 
    
    return source

def train(model, optimizer, scheduler, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches (e.g. one epoch).
    
    NOTE: If we need to check new architecture needs to run train function with limited 
          number of batched. Just to check that model able to overfit on these data.
          Here we need to decrese variable 't', e.g. t = int(t * 0.3), if want to train
          over 30% of dataset.
          
    Inputs:
    - metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch;
    - params: (Params) hyperparameters."""
    torch.cuda.benchmark=True
    
    model.train()                 # drop out and batch norm in train mode
    torch.set_grad_enabled(True)  # calculate grads     
    
    summ = []                         # summary for current train loop
    loss_avg = utils.RunningAverage() # running avg object for loss

    # use tqdm for progress bar    
    t = tqdm.tqdm(dataloader)
    for i, (images, labels) in enumerate(t): 
        images, labels = images.to(device), labels.to(device)
        
        # compute model output and loss
        scores = model(images)
        loss = loss_fn(scores, labels)
        
        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad(); 
        loss.backward()
        
        # performs updates using calculated gradients
        optimizer.step(); 
        scheduler.step()
        
        # evaluate model once per save_summary_steps
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable -> move to cpu -> numpy arrays
            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            summary_batch = {metric: metrics[metric](scores, labels) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # update the average loss
        loss_avg.update(loss.item())
        # TODO: add all metrics ploting
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()), refresh=False)
       
    # compute mean of all metrics in summary, e.g. for epoch
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    
    return {'train': metrics_mean}


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, monitor_metric, 
                       loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.
    Inputs:
    - monitor_metric: (string) name of metric in metrics dics which used for early stopping callback
    - metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    - params: (Params) hyperparameters
    - model_dir: (string) directory containing config, weights and log
    - restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)."""
    
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    
    # early stopping 
    early_stopping = params.early_stopping   
    assert type(early_stopping) == dict, "early_stopping variable should be a dict."
    callback = utils.EarlyStopping(optimizer, monitor_metric=early_stopping['monitor_metric'], 
                                   mode=early_stopping['mode'], patience=early_stopping['patience'])
    
    # init history dict 
    assert type(metrics) == dict, "Metrics variable is not a dict."
    list_of_metircs = list(metrics.keys()) + ['loss']
    history = {'train' : {k: [] for k in list_of_metircs}, 'valid': {k: [] for k in list_of_metircs}}
    
    # init visualization
    rt_monitoring = utils.RealTimeMonitoring(history)
    # allows to run script without closing figure & give a time to load empty plot
    plt.show(block=False); plt.pause(0.1)
            
    
    for epoch in range(params.num_epochs):
        # run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch and save to history
        train_metrics = train(model, optimizer, scheduler, loss_fn, train_dataloader, metrics, params)
        for metric in list_of_metircs:  history['train'][metric].append(train_metrics['train'][metric])
        
        # evaluate and write to history for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params) 
        for metric in list_of_metircs:  history['valid'][metric].append(val_metrics[metric])
            
        # plot step
        rt_monitoring.update(history)
            
        # track early stopping call back
        stop_training = callback.update(val_metrics)   
        pat = early_stopping['patience']
        if stop_training: 
            logging.info(f"- Early Stopping: val {monitor_metric} doesn't improve {pat} epochs. Stop training.")
            break  
               
        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # if best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
    
    plt.close()
    
    # save history to json in model's directory
    history_json_path = os.path.join(model_dir, "history.json")
    with open(history_json_path, 'w') as file:
        json.dump(history, file, indent=4)
        
    # save plot of model training history to the model folder
    utils.save_model_history_graphics(history, model_dir)
    

if __name__ == '__main__':
    
    # release torch memory if filled
    torch.cuda.empty_cache()

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'valid'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['valid']

    # augmentation logging
    pkl_path = os.path.join(args.model_dir, 'augm.pkl')
    data_loader.save_augmentation(data_loader.augm, pkl_path)
    logging.info("- done.")
    
    # read source file with architecture from /model_dir
    source = prepare_source_file(model_dir=args.model_dir)
    logging.info(f"Source code loaded from - {source.__file__}.")
    
    # init the model and put to cuda if available
    model = source.__dict__[f"{params.arch}"](num_classes=params.n_clas)
    model = model.to(device)
    logging.info(f"Architecture used - {params.arch}.")
    
    # inform early stopping activation status
    if params.early_stopping['monitor_metric']==False: logging.info("- early stopping disabled.")
    else: logging.info("Early stopping enabled.")
    
    # set calculate gradients for all or separate layers
    for layer in model.parameters(): layer.requires_grad = True
    for i in model.fc.parameters(): i.requires_grad = True
    
    # split model to differnet groups and assign lr for each
    optimizer = utils.get_optim(
        model=model, idx_to_split_model=[5, 8], optimizer=torch.optim.Adam, lrs=params.learning_rate, wd=params.wd,
    )    

    # fetch loss function and metrics
    loss_fn = torch.nn.CrossEntropyLoss()  
    
    # select scheduler
    if params.scheduler == "one_cycle": 
        scheduler = utils.OneCycle(
            params.num_epochs, optimizer, div_factor=params.div_factor, pct_start=params.pct_start, dl_len=len(train_dl),
            )
    elif params.scheduler == "cosine_annealing": 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dl), eta_min=1e-6)
    
    metrics = metric.metrics
    monitor_metric = params.early_stopping["monitor_metric"]

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(
        model=model, 
        train_dataloader=train_dl, 
        val_dataloader=val_dl, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        monitor_metric=monitor_metric, 
        loss_fn=loss_fn, 
        metrics=metrics, 
        params=params, 
        model_dir=args.model_dir, 
        restore_file=args.restore_file,
        )
