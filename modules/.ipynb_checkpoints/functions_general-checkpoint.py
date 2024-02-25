# import packages

# general tools
import numpy as np
import os
import glob






# general functions

def save_X_smiles_as_csv(X_smiles, location):

    indices = np.reshape(np.arange(0, len(X_smiles)), (-1,1))
    data = np.concatenate((X_smiles, indices), axis = 1)
    np.savetxt(location, data, delimiter = ",", fmt = "%s")


def save_Y_as_csv(Y, location):

    indices = np.reshape(np.arange(0, len(Y)), (-1,1))
    data = np.concatenate((Y, indices), axis = 1)
    np.savetxt(location, data, delimiter = ",")

    
def delete_all_files_in_folder(filepath):
    files = glob.glob(filepath + "*")
    for f in files:
        os.remove(f)
        
        
# functions to create neural network architecture tuples

def all_combs_list(l_1, l_2):
    
    all_combs = []
    
    for a in l_1:
        for b in l_2:
            all_combs.append((a,b))
   
    return all_combs


def arch(input_dim = 200, output_dim = 1, hidden_width = 300, hidden_depth = 10):
    
    hidden_layer_list = [hidden_width for h in range(hidden_depth)]
    arch = tuple([input_dim] + hidden_layer_list + [output_dim])
    
    return arch