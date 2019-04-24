TODO:
REWRITE THIS IN GITHUB MANNER.

- plot_model_results.py: when we train single model we need to save figure with training history in the model directiory. Afte this we can try to show a figure with real time trainig results.
- logging_grad_flow.py: logging gradients from first and last layers.
- hyper_search.py: hyperparams search

If need to overfit model that go to data_loader.py script and uncomnet line 
which limit number of samples in train dataloader.

                       
# Folders
-- notebook  :  all experiments we are executing through python scritps,
                but for prototyping we are using Jupyter Notebooks which
                is stored in this folder.

-- data      :  in our application we need to classify raw objects in the 
                way they stored and selling in supermarket. So for validation 
                and test sets we need to filter imagenet images. For 
                classification I manulay filter images. For class I leaved 
                only images with unambigious vegatables and fruits. We don't 
                use images of coocked objects. Only raw. These folder should 
                contain: train, valid, test. Each class should have separate fodler.
                
-- experiment:  for every new experiment, you will need to create a new directory 
                under experiments with a similar params.json file.

-- model     :  all layers and models. We use it in train.py file to run a model


# Scrips
-- synthesize_results.py: script which goes through all folder in --parent_dir and 
                          read best_val.json and generate file with table of these data. So
                          as result we have table with experiment name and best values for it.
                
-- plot_results.py      : script goes throuth all folders in --parent_dir and read history.json
                          and save to disk .png file with plotted curves from history files.
                       
-- include.py           : contrain all libruraries and modules which we are using.

-- evaluate.py          : scripts run model to evaluate it on test data.

-- search_hyperparams.py: used to run several experiments/models. Generate result figure and tabular.
                          Params for search should be defined in params.json dict.
                


