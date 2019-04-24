from include import *
import utils
from model import data_loader, net, metric
from evaluate import evaluate

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/imagenet/', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training") # 'best' or 'train'


class RealTimeMonitoring(object):
    """Class visualizing training in real time. Helps to monitor and 
    debug nn while training. Takes history dict updates while trainins.
    Pay attention history.json doesn't exist yet. We are using history 
    variable fro train.py script on the fly.
    
    Example:
    
    --init history = {}--
    monitor = RealTimeMonitoring(history)
    --update history--
    monitor.update(history)
    
    Inputs:
    - history: (dict) containing values of metrics for train and valid states."""
    
    def __init__(self, history):
        """Automaticaly build figure based on number of metrics. Each metric
        has it's own plot. Each plot contain two curves: train and valid."""
        
        self.metrics = list(history['train'].keys())
        num_metrics = len(self.metrics)
        self.fig, self.axes = plt.subplots(1, num_metrics, figsize=(16, 6))
        # just for visual comford after first epoch
        for ax in (self.axes.flatten()):
            ax.set_ylabel("Y-label", fontdict={'fontsize': 12}); ax.set_xlabel('X-label')
            ax.set_title("Title", fontdict={'fontsize': 12})
            
    def _animate(self, history, ax, metric):
        """Func update one axes with train and valid curves of one metric. 
        So this is one plot of one metric but two curves: train and valid.
        Inputs:
        - history: (dict) updated hsitory with last batch metrics
        - ax: (axes) where we need to plot metric for train, valid
        - metric: (scting) metric which we need to plot."""
        
        ax.clear()
        for index, (ax, state) in enumerate(zip([ax, ax], ['train', 'valid'])):
            data = history[state][ metric ]
            ax.plot(data, label=f'{state}')
            ax.set_ylabel(f'{metric}', fontdict={'fontsize': 12})
            ax.set_xlabel('Epochs'); 
            ax.set_title(f'{metric}', fontdict={'fontsize': 12})
            ax.legend()
            
    def update(self, history):
        """Call function to update the figure with plots."""
        
        # number of plots (axes) is equal to number of metrics.
        for index, ax in enumerate(self.axes.flatten()):
            animation.FuncAnimation(self.fig, self._animate(history, ax, self.metrics[index]), interval=1000)
            ax.grid(); plt.pause(0.001)
        
        
class EarlyStopping(object):
    def __init__(self, optimizer, monitor_metric=False, minimize=True, patience=5, min_delta=0.1):
        """Callback. We have two options to define criteria for stopping. First is to track delta
        between train and validation metric. When this defference encreasing that stop
        training. Second option, implemented here, is to track monitor metric and if
        this metric doesn't imporve patience epochs that stop training. We check abs difference
	and if metric doesn't change (increse or decrease, don't care) than we stop training.
	Means that we stop training when monitor metric on plato. But we can define how many epochs
	we can wait (patience).

        Inputs:
        - monitor: (string) name of validation metric to track.
        - patience: (int) number of epochs to wait after we found no improvement.
        - min_delta: (int) minimum change in the monitored quantity to qualify as 
          an improvement, i.e. an absolute change of less than min_delta, will 
          count as no improvement."""

        self.optimizer = optimizer
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.min_delta = min_delta
        
        self.metric_history = [0.001]
        self.counter = 0

    def update(self, batch_metrics):
        """Each call this func is comparing current batch metric with last on in the
        history, if delta is less than min_delta than counter + 1."""
        
        # if monitor_metric doesn't define that skip early stopping
        if self.monitor_metric == False:
            return False
        
        # unpack monitor metric from metrics
        metric = batch_metrics[self.monitor_metric]   

        # calculate delta between current batch metric and last one in history
        if np.abs(self.metric_history[-1] - metric) < self.min_delta: self.counter += 1
        else: self.counter = 0 

        # append curent batch metric to history
        self.metric_history.append(metric)

        stop_train = True if self.counter == self.patience else False
        return stop_train
    

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
    callback = EarlyStopping(optimizer, monitor_metric=early_stopping['monitor_metric'], 
                             minimize=early_stopping['minimize'], patience=early_stopping['patience'], 
                             min_delta=early_stopping['min_delta'])
    
    # init history dict 
    assert type(metrics) == dict, "Metrics variable is not a dict."
    list_of_metircs = list(metrics.keys()) + ['loss']
    history = {'train' : {k: [] for k in list_of_metircs}, 'valid': {k: [] for k in list_of_metircs}}
    
    # init visualization
    rt_monitoring = RealTimeMonitoring(history)
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
    
    # Define the model and optimizer
    if params.arch=="resnet18": model = net.resnet18(num_classes=params.n_clas, pretrained=params.pretrain,ps=params.ps)
    elif params.arch=="se_resnet18": model = net.se_resnet18(num_classes=params.n_clas, pretrained=params.pretrain, ps=params.ps)
    elif params.arch=="cbam_rn18": model = net.cbam_resnet18(num_classes=params.n_clas, pretrained=params.pretrain, ps=params.ps)
    elif params.arch=="resnet152": model = net.resnet152(num_classes=params.n_clas, pretrained=params.pretrain, ps=params.ps)
    model = model.to(device)
    logging.info(f"- architecture - {params.arch}.")
    
    # inform that early stopping disabled
    if params.early_stopping['monitor_metric']==False: logging.info("- early stopping disabled.")
    else: logging.info("- early stopping enabled.")
    
    for layer in model.parameters(): layer.requires_grad = False
    for i in model.fc.parameters(): i.requires_grad = True
    
    # split model to differnet groups and assign lr for each
    optimizer = utils.get_optim(model=model, idx_to_split_model=[5, 8], optimizer=torch.optim.Adam, 
                                lrs=params.learning_rate, wd=params.wd)    
#     optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.wd)

    # fetch loss function and metrics
    loss_fn = torch.nn.CrossEntropyLoss()  
    
    # SCHEDULER
    if params.scheduler == "one_cycle": \
        scheduler = utils.OneCycle(params.num_epochs, optimizer, div_factor=params.div_factor,   
                                   pct_start=params.pct_start, dl_len=len(train_dl))
    elif params.scheduler == "cosine_annealing": \
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dl), eta_min=1e-6)
    
    metrics = metric.metrics
    monitor_metric = "loss"

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, scheduler, monitor_metric, loss_fn, metrics, params, 
                       args.model_dir, args.restore_file)
