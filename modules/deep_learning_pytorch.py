import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .scoring import regression_scores, binary_classification_scores



def arch(input_dim = 200, output_dim = 1, hidden_width = 300, hidden_depth = 10):
    """
    Returns a tuple of integers specifying the architecture of an MLP. For example (200, 100, 100, 100, 1) specifies an MLP with input dim = 200, three hidden layers with 100 neurons each, and output dim = 1.
    """
    
    hidden_layer_list = [hidden_width for h in range(hidden_depth)]
    arch = tuple([input_dim] + hidden_layer_list + [output_dim])
    
    return arch



def all_combs_list(l_1, l_2):
    """
    Creates a list of all possible pairs (a,b) whereby a is in l_1 and b is in l_2.
    """
    
    all_combs = []
    
    for a in l_1:
        for b in l_2:
            all_combs.append((a,b))
   
    return all_combs



class MLP(nn.Module):
    """
    MLP class with variable architecture, implemented in PyTorch. Optionally includes batchnorm and dropout.
    """
    
    def __init__(self, 
                 architecture = (1, 10, 10, 1), 
                 hidden_activation = nn.ReLU(), 
                 output_activation = nn.Identity(), 
                 use_bias = True, 
                 hidden_dropout_rate = 0.0, 
                 hidden_batchnorm = False):
        
        # inherit initialisation method from parent class
        super(MLP, self).__init__()
        
        # define computational layers
        self.layers = nn.ModuleList()
        
        for k in range(len(architecture)-1):
            
            # add batchnorm layer
            if k > 0 and hidden_batchnorm == True:
                self.layers.append(nn.BatchNorm1d(architecture[k]))
            
            # add dropout layer
            if k > 0:
                self.layers.append(nn.Dropout(p = hidden_dropout_rate))
           
            # add affine-linear transformation layer
            self.layers.append(nn.Linear(architecture[k], architecture[k+1], bias = use_bias))
            
            # add nonlinear activation layer
            if k < len(architecture) - 2:
                self.layers.append(hidden_activation)
            else:
                self.layers.append(output_activation)
                
    def forward(self, x):
        
        # apply computational layers in forward pass
        for layer in self.layers:
            x = layer(x)
        
        return x

    
    
def create_mlp_model(ml_settings):
    
    mlp_model = MLP(architecture = ml_settings["architecture"], 
                    hidden_activation = ml_settings["hidden_activation"], 
                    output_activation = ml_settings["output_activation"], 
                    use_bias = ml_settings["use_bias"], 
                    hidden_dropout_rate = ml_settings["hidden_dropout_rate"], 
                    hidden_batchnorm = ml_settings["hidden_batchnorm"])
    
    return mlp_model



def fit_pytorch_mlp_model(model,
                          dataset_train,
                          dataset_test = None,
                          batch_size = 2**7,
                          dataloader_shuffle = True, 
                          dataloader_drop_last = True,
                          learning_rate = 1e-3,
                          lr_lambda = lambda epoch: 1,
                          lr_last_epoch = 0,
                          weight_decay = 1e-2,
                          num_epochs = 1,
                          loss_function = nn.MSELoss(), 
                          optimiser = torch.optim.AdamW, 
                          performance_metrics = "regression", 
                          print_results_per_epochs = 1, 
                          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Training loop for PyTorch MLP model implemented in the MLP class above. Optionally includes weight decay and learning rate decay.
    """

    # assign model to computational device
    model = model.to(device)
    
    # create dataloaders
    dataloader_train = DataLoader(dataset = dataset_train,
                                  batch_size = batch_size,
                                  shuffle = dataloader_shuffle, 
                                  drop_last = dataloader_drop_last)
    
    dataloader_train_for_eval = DataLoader(dataset = dataset_train,
                                           batch_size = len(dataset_train),
                                           shuffle = False,
                                           drop_last = False)
    
    if dataset_test != None:
        
        dataloader_test = DataLoader(dataset = dataset_test,
                                     batch_size = len(dataset_test),
                                     shuffle = False,
                                     drop_last = False)
    
    # compile optimiser and learning rate scheduler
    compiled_optimiser = optimiser(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(compiled_optimiser, lr_lambda = lr_lambda)
    
    # set learning rate scheduler state via dummy loop (in case we have trained before and want to resume training at a later epoch)
    for _ in range(lr_last_epoch):
        lr_scheduler.step()
        
    # preallocate arrays to save loss curves on training- and test set
    loss_curve_training_set = np.zeros(num_epochs)
    loss_curve_test_set = np.zeros(num_epochs)
    
    # loop over training epochs
    for epoch in range(num_epochs):

        # set model to training mode
        model.train()
        
        # loop over minibatches for training
        for (feature_vector_batch, label_vector_batch) in dataloader_train:
            
            # assign data batch to computational device
            feature_vector_batch = feature_vector_batch.to(device)
            label_vector_batch = label_vector_batch.to(device)

            # compute current value of loss function via forward pass
            loss_function_value = loss_function(model(feature_vector_batch), label_vector_batch)

            # set past gradient to zero
            compiled_optimiser.zero_grad()
            
            # compute current gradient via backward pass
            loss_function_value.backward()
            
            # update model weights using gradient and optimisation method
            compiled_optimiser.step()
        
        # apply learning rate scheduler
        lr_scheduler.step()
        
        # set model to evaluation mode
        model.eval()
        
        # generate current predictions and loss function values of model on training- and test set
        for (feature_vector_batch, label_vector_batch) in dataloader_train_for_eval:
            
            feature_vector_batch = feature_vector_batch.to(device)
            label_vector_batch = label_vector_batch.to(device)
            
            y_train_pred = model(feature_vector_batch).cpu().detach().numpy()[:,0]
            y_train_true = label_vector_batch.detach().cpu().numpy()[:,0]
        
        training_loss = loss_function(torch.tensor(y_train_true, dtype = torch.float32), torch.tensor(y_train_pred, dtype = torch.float32))
        loss_curve_training_set[epoch] = training_loss
        
        if dataset_test != None:
            
            for (feature_vector_batch, label_vector_batch) in dataloader_test:

                feature_vector_batch = feature_vector_batch.to(device)
                label_vector_batch = label_vector_batch.to(device)
                
                y_test_pred = model(feature_vector_batch).cpu().detach().numpy()[:,0]
                y_test_true = label_vector_batch.detach().cpu().numpy()[:,0]

            test_loss = loss_function(torch.tensor(y_test_true, dtype = torch.float32), torch.tensor(y_test_pred, dtype = torch.float32))
            loss_curve_test_set[epoch] = test_loss
        
        # print current performance metrics (if wanted)
        if print_results_per_epochs != None:
            if (epoch + 1) % print_results_per_epochs == 0:
                
                if performance_metrics == "regression":
                    print("Results after ", epoch + 1, " epochs on training set:")
                    regression_scores(y_train_true, y_train_pred, display_results = True)

                    if dataset_test != None:
                        print("Results after ", epoch + 1, " epochs on test set:")
                        regression_scores(y_test_true, y_test_pred, display_results = True)

                    print("\n \n")

                elif performance_metrics == "classification":
                    print("Results after ", epoch + 1, " epochs on training set:")
                    binary_classification_scores(y_train_true, y_train_pred, display_results = True)

                    if dataset_test != None:
                        print("Results after ", epoch + 1, " epochs on test set:")
                        binary_classification_scores(y_test_true, y_test_pred, display_results = True)

                    print("\n \n")

                else:
                    print("Neither regression- nor classification task.")

    return (model, num_epochs, loss_curve_training_set, loss_curve_test_set)



def train_mlp_model(mlp_model, ml_settings, X_train, y_train, X_test = None, y_test = None):
    
    dataset_train = TensorDataset(torch.tensor(X_train, dtype = torch.float), 
                                  torch.tensor(y_train.reshape(-1,1), dtype = torch.float))
    
    if X_test is not None and y_test is not None:
        
        dataset_test = TensorDataset(torch.tensor(X_test, dtype = torch.float), 
                                     torch.tensor(y_test.reshape(-1,1), dtype = torch.float))
    else:
        
        dataset_test = None

    (loss_curve_training_set, loss_curve_test_set) = fit_pytorch_mlp_model(model = mlp_model,
                                                                           dataset_train = dataset_train,
                                                                           dataset_test = dataset_test,
                                                                           batch_size = ml_settings["batch_size"],
                                                                           dataloader_shuffle = ml_settings["dataloader_shuffle"], 
                                                                           dataloader_drop_last = ml_settings["dataloader_drop_last"],
                                                                           learning_rate = ml_settings["learning_rate"],
                                                                           lr_lambda = ml_settings["lr_lambda"],
                                                                           lr_last_epoch = ml_settings["lr_last_epoch"],
                                                                           weight_decay = ml_settings["weight_decay"],
                                                                           num_epochs = ml_settings["num_epochs"],
                                                                           loss_function = ml_settings["loss_function"], 
                                                                           optimiser = ml_settings["optimiser"], 
                                                                           performance_metrics = ml_settings["performance_metrics"], 
                                                                           print_results_per_epochs = ml_settings["print_results_per_epochs"], 
                                                                           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))[2:]
                                                  
    return (loss_curve_training_set, loss_curve_test_set)
