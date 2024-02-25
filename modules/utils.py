import numpy as np
import random
import os
import glob
from rdkit.Chem import Draw
from sklearn.preprocessing import KBinsDiscretizer


def display_mol_with_atom_indices(mol):
        
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", str(atom.GetIdx()))

    image = Draw.MolToImage(mol)

    return image



def discretise(y_cont, n_bins = 2, strategy = "uniform"):
    
    discretiser = KBinsDiscretizer(n_bins = n_bins, encode = "ordinal", strategy = strategy)
    y_disc = list(discretiser.fit_transform(np.array(y_cont).reshape(-1,1)).reshape(-1).astype(int))
    
    return y_disc



def remove_random_element(my_list, random_state = 42):
    
    random.seed(random_state)
    my_list.pop(random.randrange(len(my_list)))
    
    return my_list



def delete_all_files_in_folder(filepath):
    files = glob.glob(filepath + "*")
    for f in files:
        os.remove(f)