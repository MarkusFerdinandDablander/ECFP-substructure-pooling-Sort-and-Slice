import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .scoring import regression_scores, binary_classification_scores



def arch(input_dim = 200, output_dim = 1, hidden_width = 300, hidden_depth = 10):
    """
    Returns a tuple of integers specifying the architecture of an MLP. For example, the tuple (200, 100, 100, 100, 1) specifies an MLP with input_dim = 200, output_dim = 1, hidden_width = 100, hidden_depth = 3.
    """
    
    hidden_layer_list = [hidden_width for h in range(hidden_depth)]
    arch = tuple([input_dim] + hidden_layer_list + [output_dim])
    
    return arch



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



def train_mlp_model(mlp_model, 
                    ml_settings, 
                    X_train, 
                    y_train, 
                    X_test = None, 
                    y_test = None, 
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    """
    Training loop for PyTorch MLP model implemented in the MLP class above.
    
    (X_train, y_train) are two numpy arrars (2d and 1d respectively) that specify the training data.
    (X_test, y_test) are two optional numpy arrars (2d and 1d respectively) that specify the test data.
    
    
    Training options that can be specified in the dictionary ml_settings:
    
    ml_settings["batch_size"],
    ml_settings["dataloader_shuffle"], 
    ml_settings["dataloader_drop_last"],
    ml_settings["learning_rate"],
    ml_settings["lr_lambda"],
    ml_settings["lr_last_epoch"],
    ml_settings["weight_decay"],
    ml_settings["num_epochs"],
    ml_settings["loss_function"], 
    ml_settings["optimiser"], 
    ml_settings["performance_metrics"], 
    ml_settings["print_results_per_epochs"].
    """
    
    # create training data set
    dataset_train = TensorDataset(torch.tensor(X_train, dtype = torch.float), 
                                  torch.tensor(y_train.reshape(-1,1), dtype = torch.float))
    
    # if available, create test data set
    if X_test is not None and y_test is not None:
        
        dataset_test = TensorDataset(torch.tensor(X_test, dtype = torch.float), 
                                     torch.tensor(y_test.reshape(-1,1), dtype = torch.float))
    else:
        
        dataset_test = None
        
    # assign mlp_model to computational device
    mlp_model = mlp_model.to(device)
    
    # create dataloaders
    dataloader_train = DataLoader(dataset = dataset_train,
                                  batch_size = ml_settings["batch_size"],
                                  shuffle = ml_settings["dataloader_shuffle"], 
                                  drop_last = ml_settings["dataloader_drop_last"])
    
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
    compiled_optimiser = ml_settings["optimiser"](mlp_model.parameters(), lr = ml_settings["learning_rate"], weight_decay = ml_settings["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(compiled_optimiser, lr_lambda = ml_settings["lr_lambda"])
    
    # set learning rate scheduler state via dummy loop (in case we have trained before and want to resume training at a later epoch)
    for _ in range(ml_settings["lr_last_epoch"]):
        lr_scheduler.step()
        
    # preallocate arrays to save loss curves on training- and test set
    loss_curve_training_set = np.zeros(ml_settings["num_epochs"])
    loss_curve_test_set = np.zeros(ml_settings["num_epochs"])
    
    # loop over training epochs
    for epoch in range(ml_settings["num_epochs"]):

        # set mlp_model to training mode
        mlp_model.train()
        
        # loop over minibatches for training
        for (feature_vector_batch, label_vector_batch) in dataloader_train:
            
            # assign data batch to computational device
            feature_vector_batch = feature_vector_batch.to(device)
            label_vector_batch = label_vector_batch.to(device)

            # compute current value of loss function via forward pass
            loss_function_value = ml_settings["loss_function"](mlp_model(feature_vector_batch), label_vector_batch)

            # set past gradient to zero
            compiled_optimiser.zero_grad()
            
            # compute current gradient via backward pass
            loss_function_value.backward()
            
            # update mlp_model weights using gradient and optimisation method
            compiled_optimiser.step()
        
        # apply learning rate scheduler
        lr_scheduler.step()
        
        # set mlp_model to evaluation mode
        mlp_model.eval()
        
        # generate current predictions and loss function values of mlp_model on training- and test set
        for (feature_vector_batch, label_vector_batch) in dataloader_train_for_eval:
            
            feature_vector_batch = feature_vector_batch.to(device)
            label_vector_batch = label_vector_batch.to(device)
            
            y_train_pred = mlp_model(feature_vector_batch).cpu().detach().numpy()[:,0]
            y_train_true = label_vector_batch.detach().cpu().numpy()[:,0]
        
        training_loss = ml_settings["loss_function"](torch.tensor(y_train_true, dtype = torch.float32), torch.tensor(y_train_pred, dtype = torch.float32))
        loss_curve_training_set[epoch] = training_loss
        
        if dataset_test != None:
            
            for (feature_vector_batch, label_vector_batch) in dataloader_test:

                feature_vector_batch = feature_vector_batch.to(device)
                label_vector_batch = label_vector_batch.to(device)
                
                y_test_pred = mlp_model(feature_vector_batch).cpu().detach().numpy()[:,0]
                y_test_true = label_vector_batch.detach().cpu().numpy()[:,0]

            test_loss = ml_settings["loss_function"](torch.tensor(y_test_true, dtype = torch.float32), torch.tensor(y_test_pred, dtype = torch.float32))
            loss_curve_test_set[epoch] = test_loss
        
        # print current performance metrics (if wanted)
        if ml_settings["print_results_per_epochs"] != None:
            if (epoch + 1) % ml_settings["print_results_per_epochs"] == 0:
                
                if ml_settings["performance_metrics"] == "regression":
                    print("Results after ", epoch + 1, " epochs on training set:")
                    regression_scores(y_train_true, y_train_pred, display_results = True)

                    if dataset_test != None:
                        print("Results after ", epoch + 1, " epochs on test set:")
                        regression_scores(y_test_true, y_test_pred, display_results = True)

                    print("\n \n")

                elif ml_settings["performance_metrics"] == "classification":
                    print("Results after ", epoch + 1, " epochs on training set:")
                    binary_classification_scores(y_train_true, y_train_pred, display_results = True)

                    if dataset_test != None:
                        print("Results after ", epoch + 1, " epochs on test set:")
                        binary_classification_scores(y_test_true, y_test_pred, display_results = True)

                    print("\n \n")

                else:
                    print("Neither regression- nor classification task.")
    
    return (loss_curve_training_set, loss_curve_test_set)



def make_mlp_prediction(mlp_model, X):
    
    """
    Make predictions via an MLP model based on a 2d numpy array whose rows are input feature vectors.
    """
    
    preds = mlp_model(torch.tensor(X, dtype = torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')).cpu().detach().numpy()[:,0]
    
    return preds