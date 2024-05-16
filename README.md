# Sort & Slice: A Simple and Superior Alternative to Hash-Based Folding for Extended-Connectivity Fingerprints

Code repository for the paper Sort & Slice: A Simple and Superior Alternative to Hash-Based Folding for Extended-Connectivity Fingerprints: 

https://arxiv.org/abs/2403.17954

This repository contains:

* A simple, self-contained and fast [function](sort_and_slice_ecfp_featuriser.py) to transform RDKit mol objects into vectorial extended-connectivity fingerprints (ECFPs) pooled via Sort & Slice.
* The [code base](modules) and [data sets](data) to fully reproduce the computational results from the paper.
* [Original numerical results](results) from the experiments conducted in the paper.



## Easily Generating Vectorial ECFPs via Sort & Slice

The function in [sort_and_slice_ecfp_featuriser.py](sort_and_slice_ecfp_featuriser.py) represents a computationally efficient, easy-to-use and self-contained method to create a featuriser that can transform RDKit mol objects into vectorial ECFPs pooled via Sort & Slice (rather than via classical hash-based folding). It only relies on RDKit and NumPy and can be readily employed for molecular feature extraction and other ECFP-based applications. This function should be all you need in case you want to use vectorial Sort & Slice ECFPs in your own project. Our numerical observations suggest that ECFPs vectorised via Sort & Slice regularly (and sometimes substantially) outperform ECFPs vectorised via classical hash-based folding across a wide variety of molecular property prediction scenarios.

EXAMPLE:
    
First use a training set of RDKit mol objects 

    [mol_1, mol_2, ...] 
    
to construct a molecular featurisation function with desired ECFP hyperparameter settings via
    
    ecfp_featuriser = construct_sort_and_slice_ecfp_featuriser(mols_train = [mol_1, mol_2, ...], 
                                                               max_radius = 2, 
                                                               pharm_atom_invs = False, 
                                                               bond_invs = True, 
                                                               chirality = False, 
                                                               sub_counts = True, 
                                                               vec_dimension = 1024, 
                                                               break_ties_with = lambda sub_id: sub_id, 
                                                               print_train_set_info = True)
                                                               
Then ecfp_featuriser(mol) is a 1-dimensional numpy array of length vec_dimension representing the vectorial ECFP for mol pooled via Sort & Slice. The function ecfp_featuriser can be thought of as
1. first generating the (multi)set of integer ECFP-substructure identifiers for mol and then
2. vectorising it via a Sort & Slice operator trained on [mol_1, mol_2, ...] (rather than vectorising it via classical hash-based folding).



## Computational Experiments

The computational results reported in the paper can be reproduced and visualised by running the Jupyter notebook [substructure_pooling_experiments.ipynb](substructure_pooling_experiments.ipynb) which uses the code base in [modules](modules). The [data folder](data) contains the five (cleaned) chemical data sets for the diverse prediction tasks investigated in the paper: lipophilicity, aqueous solubility, SARS-CoV-2 main protease inhibition, mutagenicity, and estrogen receptor alpha antagonism. Each data set is given as a set of labelled SMILES strings. The computational environment in which the original results were conducted can be found in [environment.yml](environment.yml). The original numerical results from the paper are saved in [results](results).
 
![Substructure Pooling Overview](/figures/sub_pool_methods_overview.png)

