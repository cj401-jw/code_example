# TODO: Rewrite visualization functions to class

from include import *


class RunningAverage():
    """A simple class that maintains the running average of a quantity.
        
    Example:
    
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3 """
    
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
    
    
class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params."""

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)     
            self.__dict__.update(params)

    def save(self, json_path):
        """Save existing dict to json_path"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Here is a magic. It's allow to get access to dict keys just like as method.
        Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    
    logging.info("Starting training...")
    
    Inputs:
    - log_path: (string) where to log."""
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file.
    Inputs:
    - d: (dict) of float-castable values (np.float, int, float, etc.)
    - json_path: (string) path to json file."""
    
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)    
        
        
def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
#     else:
#         print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
        

def random_split_df(df, valid_size=0.2):
    """Функция шафлит df и берет первые по порядку 
    для обучающей выборки и последние valid_size 
    для валидационной выборки. Возвращает два датафрейма
    с новыми индексами по порядку, потому что для 
    датасета они нужны по порядку. То есть делает 
    reset_index().
    
    Inputs:
    - ds: dataframe.
    - valid_size: size of validation set.
    
    Returns:
    - train: the same structure dataframe.
    - valid: the same structure dataframe."""
    
    valid_idx = int(len(df)*valid_size)
    
    # shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    train = df.iloc[ :-valid_idx,  :]
    valid = df.iloc[ -valid_idx:,  :]
    
    # reset indexing in output dfs
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    return train, valid


def trainer(model, optimizer, scheduler, loss_function, 
            epochs, training_data, validation_data, save_best=False):
    """It's my own trainer function. 
    This function is apropriate for image classification 
    task. History write only epochs."""
    
    # should be init automat
    history = {  
        "train": {"lr": [], "lr1": [], "lr2": [], "betas": [], "betas1": [], "betas2": [], "loss": [], "metric": []},
        "valid": {"loss": [], "lr": [], "metric": []}}
    
    for epoch in range(epochs):
        training_loss = validation_loss = max_metric = 0.0
        alpha = 0.8
        
        # loop over training and validation for each epoch
        for dataset, training in [(training_data, True), (validation_data, False)]:
            correct = total = 0
            torch.set_grad_enabled(training)
            model.train(training)
            t = tqdm.tqdm_notebook(dataset)
            batch_len = len(t) - 1 
        
            # loop over dataset
            for batch_idx, (images, labels) in enumerate(t):               
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                scores = model(images)
                loss = loss_function(scores, labels)
                
                # calculate metrics
                predictions = torch.argmax(scores, dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels) 
                accuracy = round(correct / total, 3)
                
                # do all stuff for train and validation
                if training:
                    loss.backward()
                    # calc moving average loss
                    if batch_idx==0: training_loss = loss.item()
                    training_loss_ = loss.item()
                    training_loss = (alpha*training_loss) + ((1-alpha)*training_loss_)
                    t.set_postfix(epoch=epoch, training_loss=training_loss,
                            accuracy=accuracy, refresh=False)
                    
                    # track history
                    history['train']['lr'].append(optimizer.param_groups[0]['lr'])
#                     history['train']['lr1'].append(optimizer.param_groups[1]['lr'])
#                     history['train']['lr2'].append(optimizer.param_groups[2]['lr'])
                    
                    history['train']['betas'].append(optimizer.param_groups[0]['betas'])
#                     history['train']['betas1'].append(optimizer.param_groups[1]['betas'])
#                     history['train']['betas2'].append(optimizer.param_groups[2]['betas'])
                    
                    history['train']['loss'].append(training_loss)
                    history['train']['metric'].append(accuracy)
                    # update weights
                    optimizer.step()
                    scheduler.step()
                else:
                    validation_loss = loss.item()       
                    t.set_postfix(epoch=epoch, validation_loss=validation_loss,
                            accuracy=accuracy, refresh=False)
                    # track history
                    history['valid']['loss'].append(validation_loss)
                    history['valid']['metric'].append(accuracy)
                    # save history at the epoch's end
                    if save_best and batch_idx == batch_len and accuracy > max_metric:
                        max_metric = accuracy
                        torch.save(model.state_dict(), "BEST_WEIGHTS.pt") 
                        print(f"Best model with accuracy: {round(accuracy, 3)} saved to BEST_WEIGHTS.pt")
    return history


# ============================================
# Visualization utils
# ============================================


# TODO: Rewrite input args to apropriate vars
def check_augm_show_images(args, rows=3, cols=4):
    """For simple classification task. To check the 
    augmentation we are ploting images with augm."""
    
    # check that we have enough images in batch
    assert rows*cols <= args["batch_size"], \
    "You're trying to plot more images then batch_size."
        
    images, labels = next(iter(args["dataloaders"]['train']))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, 
                             figsize=(14, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(reverse_transform(images[i]))
        ax.axis('off')
    plt.tight_layout()
    
    
def show_lr_and_moms(history):
    """Plot learning rate and momentum. We do this 
    to check learning schem. How lr and momentum 
    are changing."""
    
    # TODO: add check that structure of history 
    # variable is appropriate and has necesery keys.
      
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train']['lr']);
    axes[0].set_title('Learning rate changes over training');
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Learning rate')
    
    axes[1].plot(history['train']['betas']);
    axes[1].set_title('Momentum changes over training');
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Momentum')
    
    
def moving_average(sequence, alpha=0.999):
    """This function takes input sequence and average."""
    
    avg_loss = sequence[0]
    average = []
    for n, o in enumerate(sequence):
        avg_loss = (alpha*avg_loss) + ((1-alpha)*o)
        average.append(avg_loss)
    return average


def reverse_transform(image: torch.Tensor) -> np.ndarray:
    '''Convert tensor on which performed ToTensor, 
    Normilization torch transformation back to numpy 
    array appropriate for plotting with matplotlib.
    Does handle cuda and cpu tensors.
    
    Inputs:
    - image: torch.Tensor with shape [C x H x W].
      Normilized with imagenet_stats.
      
    Returns:
    - image_np: np.ndarray with size [H x W x C].
      Range of values between 0, 255. Appropriate
      for plotting with matplotlib.'''
    
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_np = image.cpu().numpy().transpose((1, 2, 0))
    image_np = 255 * np.clip( 
        (std * image_np + mean), a_min=0, a_max=1 )
    return image_np.astype(np.uint8)


def show_train_results(history : dict):
    """Plot training history.
    Inputs:
    - history: dict with keys: train, valid with 
      results of training. 
    - step: lenght of train and valid sets not 
      equal so we plot train history with step 
      gaps. Validation history plots fully."""
    
    # We need this since train and val lengh is different
    step = len(history['train']['metric']) // len(history['valid']['metric'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(history['train']['loss'][::step], label='train')
    axes[0].plot(history['valid']['loss'], label='valid')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history['train']['metric'][::step], label='train')
    axes[1].plot(history['valid']['metric'], label='valid')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Metric')
    axes[1].legend()
    plt.tight_layout()


# ============================================
# Segmentation
# ============================================
def show_segment_batch(images: torch.Tensor, masks: torch.Tensor, 
                       how_many_plot: int, alpha=0.5):    
    """Function which plots images and masks batches
    returned from torch.Dataloader. Retruns nothing  
    but plot images with masks. Always plots 3 columns. 
    Rows only vary. Very important to set N in cmap
    which means number of colors. Also vmin and vmax
    in imshow of mask to set scale of integers in mask.
    
    Inputs:
    - images: torch.Tensor with shape [N x CH x H x W].
      Images normalized in accordance with imagenet, 
      where N is a batch size and CH is number of 
      channles, three for RGB images.
    - masks: torch.Tensor with shape [N x C x H x W]
      where C is a number of classes. Separate binary 
      mask for each class.
    - how_many_plot: number of images to plot. Takes
      from the start, not random."""
    
    # init some variables
    class_to_color = {"background": 'black',
                  "orange": 'orange',
                  "banana": 'khaki',
                  "tomato": 'red',
                  "apple": 'green',
                  "lemon": 'yellow'}
    int_to_color = {0: "black", 
                    1: "orange", 
                    2: "khaki", 
                    3: "red", 
                    4: "green", 
                    5: "yellow"}
    _nrows = 3
    cmap = colors.ListedColormap(list(int_to_color.values()), N=6)
    
    fig, axes = plt.subplots(nrows=int(np.ceil(how_many_plot/_nrows)), 
                             ncols=_nrows, figsize=(13, 9))
    for i, ax in enumerate(axes.flatten()):
        if i > how_many_plot-1: return 
        ax.imshow( reverse_transform(images[i]) )
        uniq_classes_in_mask = sorted(masks[i].unique().cpu().numpy())
        mask = torch.squeeze(masks[i], dim=0).cpu().numpy().astype(np.uint8)
        ax.imshow( mask, vmin=0, vmax=5, alpha=alpha, cmap=cmap)
        ax.axis('off')
    plt.tight_layout()
    
    
# ============================================
# Fit One CYcle
# ============================================
class MyScheduler(object):
    """Updated base scheduler class to be appropriate to my OneCycle."""
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`. It contains an entry for every 
        variable in self.__dict__ which is not the optimizer."""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Inputs:
        - state_dict (dict): scheduler state. Should be an object returned from a call 
          to :meth:`state_dict`."""
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr, mom in zip(self.optimizer.param_groups, self.get_lr(), self.get_moms()):
            param_group['lr'] = lr
            param_group['betas'] = (mom, 0.99)
            
            
class OneCycle(MyScheduler):
    def __init__(self, epochs, optimizer, div_factor, pct_start, dl_len, last_epoch=-1):
        """Суть в том, что эта схема обучения применяеться для всех
        эпох, а не для каждой по отдельности. То есть эта общая схем для всех
        эпох. То есть каждая эпоха не имеет свой отдельный цыкл! Один цыкл для 
        всех эпох. Важно для понимания."""
        # uppack all param groups
        lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.max_lr = lrs[0]
        self.eta_min = self.max_lr / div_factor
        self.num_iterations = dl_len * epochs
        self.upward_steps = int(self.num_iterations * pct_start)
        self.max_moms=0.95
        self.min_moms=0.85
        super(OneCycle, self).__init__(optimizer, last_epoch)
        
    def _calculate_schedule(self, lr):
        """Calculate one cycle policy curves for learning rate and momentum for 
        each param group in optimizer. Calculating curves once.
        
        TODO: Currently we calculate these curves each time we call step func.
        But we can calculate these curves only once and them call reqauired value
        instead of calculating them each time.
        
        Inputs:
        - lr: (float) maximum learning rate in cycle. We defined it using learning
          rate search algorithm."""
        
        # calculate one cycle for learning rate
        upward_lr = np.linspace(start=self.eta_min, stop=lr, num=self.upward_steps)
        downward_lr = [(self.eta_min + (lr - self.eta_min) * (1 + math.cos((math.pi*o)/self.num_iterations)) / 2) 
                for o in np.linspace(start=0, stop=self.num_iterations, num=self.num_iterations-self.upward_steps)]
        
        # calculate one cycle for momentum
        upward_moms = np.linspace(start=self.max_moms, stop=self.min_moms, num=self.upward_steps)
        downward_moms = [(self.min_moms + (self.max_moms - self.min_moms) * (1 + math.cos((math.pi*o)/self.num_iterations)) / 2) 
                for o in np.linspace(start=self.num_iterations, stop=0, num=self.num_iterations-self.upward_steps)]
        
        return [np.concatenate([upward_lr, downward_lr]), np.concatenate([upward_moms, downward_moms])]
        
    def get_lr(self):
        """As said before in the above func we call self.last_epoch value to get required last value.
        The same issue with calculating momentums."""
        lr = [self._calculate_schedule(base_lr)[0][self.last_epoch] for base_lr in self.base_lrs]
        return lr
    
    def get_moms(self):
        # returns updated mom for each param group
        moms = [self._calculate_schedule(base_lr)[1][self.last_epoch] for base_lr in self.base_lrs]
        return moms
    
   
    
def split_model_idx(model, idxs):
    """Split `model` according to the indexes in `idxs`.
    
    Example: 
    
    len(model)=10; idxs=[5, 8]
    [5, 8] -> [0, 5, 8, 10] -> [0-5, 5-8, 8-10]
    
    remember last value in indexing doesn't count."""
    
    assert type(idxs)==list, 'idxs should be list of integers.'
    layers = list(model.children())
    if idxs[0] != 0: idxs = [0] + idxs
    if idxs[-1] != len(layers): idxs.append(len(layers))
    return [nn.Sequential(*layers[i:j]) for i,j in zip(idxs[:-1],idxs[1:])]


def create_opt(optimizer, group_layers, lrs, wd):
    """Assign lrs to group layer."""
    assert len(group_layers)==len(lrs), 'Len of groups should be equal to len of lrs.'
    return optimizer([{'params': x.parameters(), 'lr': lr} for x, lr in zip(group_layers, lrs)], weight_decay=wd)


def get_optim(model, idx_to_split_model, optimizer, lrs, wd):
    """Combine two above functions.
    Inpust:
    - model: (torch.nn.Module)
    - idx_to_split_model: (list) integer in list used to split model to group layers.
    - lrs: (list) list of learnings rates for each group layer, e.g. [0.1, 0.001, 0.0001]."""
    group_layers = split_model_idx(model, idx_to_split_model)
    optimizer = create_opt(optimizer, group_layers, lrs, wd)
    return optimizer


def save_model_history_graphics(history, model_dir):
    """Func is used to show single model training results. 
    Could be used to track overfitting or underfiting, 
    learning rate inconsistent or else. Num plots equal 
    to num_metrics. TODO: assert history.
    Inputs:
    - history: (dict) containing history of train.
    - model_dir: (string) model dir."""
    
    # unpach list of metrics from dict
    metrics = list(history['train'].keys())
    num_metircs = len(metrics)
    
    fig, axes = plt.subplots(1, num_metircs, figsize=(12, 6))
    for index, ax in enumerate(axes.flatten()):
        ax.plot(history['train'][ metrics[index] ], label='train')
        ax.plot(history['valid'][ metrics[index] ], label='valid')
        ax.set_ylabel(f'{metrics[index]}', fontdict={'fontsize': 14})
        ax.set_xlabel('Epochs'); ax.grid(); ax.legend()
        ax.set_title(f'{metrics[index]}', fontdict={'fontsize': 14})
    plt.tight_layout()
    
    # save figure on the dist
    figure_file_name = f'{model_dir}/training_history.png'
    plt.savefig(figure_file_name) 
    
    
def find_lr(net, dataloader, optimizer, loss_function, init_value=1e-7, final_value=10., beta=0.95):
    """https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html#how-do-you-find-a-good-learning-rate.
    We plot exponentially weighted average loss. To stop a algorighm 
    when loss exploted we define formula. We take 0 index from params 
    group for searching.
    Fast ai do the next for tune-tuning. Find lr for train classifier.
    Than find another lr for first group and for classifier use lr from 
    first stage / 10.
    
    Example: 
    
    log_lrs, losses = find_lr(model, dls['train'], optimizer, loss_function, beta=0.95)
    plot_lr_find(log_lrs, losses)
    
    Inputs:
    - beta: (float) used for loss smoothing."""
    
    num = len(dataloader)-1   # number of iterations
    lr = init_value           # initilize learning rate
    mult = (final_value / init_value) ** (1/num)  # multiplication factor to increase lr
    optimizer.param_groups[0]['lr'] = lr          # take first param group
    avg_loss, best_loss = 0., 0.
    losses, log_lrs = [], []
     
    for batch_num, (inputs, labels) in tqdm.tqdm_notebook(enumerate(dataloader, start=1)):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # get scores
        optimizer.zero_grad()
        outputs = net(inputs)
        
        # calculate and smooth loss
        loss = loss_function(outputs, labels)
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        
        # stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        
        # record the best loss, why we need this?
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
            
        # store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log(lr))
        
        # do the optimizer step
        loss.backward()
        optimizer.step()
        
        # increase the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        
    return log_lrs, losses


def plot_lr_find(log_lrs, losses):
    _, ax = plt.subplots(1,1)
    ax.plot(np.exp(log_lrs)[10:-4],losses[10:-4]);
    ax.set_xscale('log');
    ax.set_xlabel('learning rate');
    ax.set_ylabel('loss');
    
    
    

