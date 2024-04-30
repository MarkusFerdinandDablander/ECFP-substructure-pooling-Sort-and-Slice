import numpy as np
import os
import glob
import random
from collections import defaultdict
from itertools import chain
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold, StratifiedKFold



def create_data_split_dict_random(x_smiles,
                                  k_splits,
                                  m_reps,
                                  random_state_cv = 42):
    
    data_split_dict = {}
    
    for m in range(m_reps):
        for (k, (ind_train, ind_test)) in enumerate(KFold(n_splits = k_splits, 
                                                          shuffle = True, 
                                                          random_state = random_state_cv + m).split(x_smiles)):
            
            # add index data structure to dictionary
            data_split_dict[(m, k)] = (ind_train, ind_test)
            
    return data_split_dict
    
    
    
def create_data_split_dict_random_strat(x_smiles,
                                        y,
                                        k_splits,
                                        m_reps,
                                        random_state_cv = 42):
    
    data_split_dict = {}
    
    for m in range(m_reps):
        for (k, (ind_train, ind_test)) in enumerate(StratifiedKFold(n_splits = k_splits, 
                                                                    shuffle = True, 
                                                                    random_state = random_state_cv + m).split(x_smiles, y)):
            
            # add index data structure to dictionary
            data_split_dict[(m, k)] = (ind_train, ind_test)
            
    return data_split_dict



def get_ordered_scaffold_sets(x_smiles, scaffold_func = "Bemis_Murcko_generic", random_order = False, random_state = 42):

    """ Taken from https://lifesci.dgl.ai/_modules/dgllife/utils/splitters.html
    and then modified by Markus Ferdinand Dablander, doctoral student at Mathematical Institute, 
    University of Oxford.

    Group molecules based on their Bemis-Murcko scaffolds and
    order these groups based on their sizes.

    The order is decided by comparing the size of groups, where groups with a larger size
    are placed before the ones with a smaller size.

    Parameters
    ----------
    x_smiles : list or 1d np.array of of smiles strings corresponding to molecules which
        will be converted to rdkit mol objects.
    scaffold_func : str
        The function to use for computing Bemis-Murcko scaffolds. If scaffold_func = "Bemis_Murcko_generic",
        then we use first rdkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol and then apply
        rdkit.Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric to the result. The result is a
        scaffold which ignores atom types and bond orders.
        If scaffold_func = "Bemis_Murcko_atom_bond_sensitive", we only use
        dkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol and get scaffolds
        which are sensitive to atom types and bond orders.

    Returns
    -------
    scaffold_sets : list
        Each element of the list is a list of int,
        representing the indices of compounds with a same scaffold.
    """

    assert scaffold_func in ['Bemis_Murcko_generic', 'Bemis_Murcko_atom_bond_sensitive'], \
        "Expect scaffold_func to be 'Bemis_Murcko_generic' or 'Bemis_Murcko_atom_bond_sensitive', " \
        "got '{}'".format(scaffold_func)

    x_smiles = list(x_smiles)
    molecules = [Chem.MolFromSmiles(smiles) for smiles in x_smiles]
    scaffolds = defaultdict(list)

    for i, mol in enumerate(molecules):

        # for mols that have not been sanitized, we need to compute their ring information
        try:
            Chem.rdmolops.FastFindRings(mol)
            if scaffold_func == 'Bemis_Murcko_generic':
                mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                mol_scaffold_generic = MurckoScaffold.MakeScaffoldGeneric(mol_scaffold)
                smiles_scaffold = Chem.CanonSmiles(Chem.MolToSmiles(mol_scaffold_generic))
            if scaffold_func == 'Bemis_Murcko_atom_bond_sensitive':
                mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                smiles_scaffold = Chem.CanonSmiles(Chem.MolToSmiles(mol_scaffold))
            # Group molecules that have the same scaffold
            scaffolds[smiles_scaffold].append(i)
        except:
            print('Failed to compute the scaffold for molecule {:d} '
                  'and it will be excluded.'.format(i + 1))

    # order groups of molecules by first comparing the size of groups
    # and then the index of the first compound in the group.
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
    
    if random_order == True:
        random.seed(random_state)
        random.shuffle(scaffold_sets)
    
    return scaffold_sets



def number_of_scaffolds(x_smiles, scaffold_func = "Bemis_Murcko_generic"):

    """Get number of distinct scaffolds in the molecular data set x_smiles"""

    scaffold_sets = get_ordered_scaffold_sets(x_smiles = x_smiles, scaffold_func = scaffold_func)
    n_scaffolds = len(scaffold_sets)

    return n_scaffolds



def k_fold_cross_validation_scaffold_split(x_smiles,
                                           k_splits = 5,
                                           scaffold_func = "Bemis_Murcko_generic", 
                                           random_state = 42):

    """This function was taken from https://lifesci.dgl.ai/_modules/dgllife/utils/splitters.html
    and then modified by Markus Ferdinand Dablander, doctoral student at University of Oxford.

    Group molecules based on their scaffolds and sort groups based on their sizes.
    The groups are then split for k-fold cross validation.

    Same as usual k-fold splitting methods, each molecule will appear only once
    in the test set among all folds. In addition, this method ensures that
    molecules with a same scaffold will be collectively in either the training
    set or the test set for each fold.

    Note that the folds can be highly imbalanced depending on the
    scaffold distribution in the dataset.

    Parameters
    ----------
    x_smiles
        List of smiles strings for molecules which are to be transformed to rdkit mol objects.
    k_splits : int
        Number of folds to use and should be no smaller than 2. Default to be 5.
    scaffold_func : str
        The function to use for computing Bemis-Murcko scaffolds. If scaffold_func = "Bemis_Murcko_generic",
        then we use first rdkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol and then apply
        rdkit.Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric to the result. The result is a
        scaffold which ignores atom types and bond orders.
        If scaffold_func = "Bemis_Murcko_atom_bond_sensitive", we only use
        dkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol and get scaffolds
        which are sensitive to atom types and bond orders.

    Returns
    -------
    list of 2-tuples
        Each element of the list represents a fold and is a 2-tuple (ind_train, ind_test) which represent indices
        for training/testing for each fold.
    """

    assert k_splits >= 2, 'Expect the number of folds to be no smaller than 2, got {:d}'.format(k_splits)
    x_smiles = list(x_smiles)

    scaffold_sets = get_ordered_scaffold_sets(x_smiles, scaffold_func = scaffold_func, random_order = True, random_state = random_state)

    # k_splits buckets (i.e. chemical compound clusters) that form a relatively balanced partition of the dataset
    index_buckets = [[] for _ in range(k_splits)]
    for group_indices in scaffold_sets:
        bucket_chosen = int(np.argmin([len(bucket) for bucket in index_buckets]))
        index_buckets[bucket_chosen].extend(group_indices)

    k_fold_cross_validation_index_set_pairs = []
    for i in range(k_splits):
        ind_train = list(chain.from_iterable(index_buckets[:i] + index_buckets[i + 1:]))
        ind_test = index_buckets[i]
        k_fold_cross_validation_index_set_pairs.append((ind_train, ind_test))

    return k_fold_cross_validation_index_set_pairs



def create_data_split_dict_scaffold(x_smiles,
                                    k_splits,
                                    m_reps,
                                    scaffold_func = 'Bemis_Murcko_generic',
                                    random_state_cv = 42):
    
    data_split_dict = {}
    
    for m in range(m_reps):
        
            
        index_set_pairs = k_fold_cross_validation_scaffold_split(x_smiles,
                                                                 k_splits = k_splits,
                                                                 scaffold_func = scaffold_func, 
                                                                 random_state = 42 + m)
        
        for (k, (ind_train, ind_test)) in enumerate(index_set_pairs):

            # add index data structure to dictionary
            data_split_dict[(m, k)] = (ind_train, ind_test)
            
    return data_split_dict













    
    

    
    
    






