# Sort & Slice: A Simple and Superior Alternative to Hash-Based Folding for Extended-Connectivity Fingerprints

Code repository for the paper [Sort & Slice: A Simple and Superior Alternative to Hash-Based Folding for Extended-Connectivity Fingerprints](https://arxiv.org/abs/2403.17954).

This repository contains:

* A simple, self-contained and fast [function](sort_and_slice_ecfp_featuriser.py) to transform RDKit mol objects into vectorial extended-connectivity fingerprints (ECFPs) via Sort & Slice.
* The [code base](modules) and [data sets](data) to fully reproduce the computational results from the paper.
* [Original numerical results](results_original) from the experiments conducted in the paper.



## Easily Generating Vectorial ECFPs via Sort & Slice from RDKit Mol Objects

* The function in [sort_and_slice_ecfp_featuriser.py](sort_and_slice_ecfp_featuriser.py) constitutes a computationally efficient, easy-to-use and self-contained method to create a featuriser that can transform RDKit mol objects into vectorial ECFPs based on substructure pooling via Sort & Slice (rather than via classical hash-based folding).
* It only relies on RDKit and NumPy and can be readily employed for molecular feature extraction and other ECFP-based applications.
* This function should be all you need in case you want to employ vectorial Sort & Slice ECFPs.
* An extensive series of [strict computational experiments](https://arxiv.org/abs/2403.17954) indicates that ECFPs pooled via Sort & Slice regularly lead to higher (and sometimes substantially higher) predictive performance than ECFPs pooled via classical hash-based folding across a wide variety of molecular property prediction scenarios.


EXAMPLE:
    
First select a training set of RDKit mol objects 

    [mol_1, mol_2, ...] 
    
that should be used to calibrate the Sort & Slice operator. This training set can then be employed along with a set of desired ECFP hyperparameter settings to construct a molecular featurisation function:
    
    ecfp_featuriser = construct_sort_and_slice_ecfp_featuriser(mols_train = [mol_1, mol_2, ...], 
                                                               max_radius = 2, 
                                                               pharm_atom_invs = False, 
                                                               bond_invs = True, 
                                                               chirality = False, 
                                                               sub_counts = True, 
                                                               vec_dimension = 1024)
                                                               
Then ecfp_featuriser(mol) is a 1-dimensional numpy array of length vec_dimension representing the vectorial ECFP for mol pooled via a Sort & Slice operator calibrated on the training set [mol_1, mol_2, ...]. 

More specifically, the function ecfp_featuriser can be thought of as

1. first generating the (multi)set of integer ECFP-substructure identifiers for mol based on the ECFP hyperparameters (max_radius, pharm_atom_invs, bond_invs, chirality, sub_counts) and then
2. vectorising this (multi)set via a Sort & Slice operator trained on [mol_1, mol_2, ...] with output dimension vec_dimension (rather than vectorising it via classical hash-based folding).

To turn a list of RDKit mol objects mols_list into a feature matrix X whose rows correspond to Sort & Slice ECFPs one can simply run
    
    X = np.array([ecfp_featuriser(mol) for mol in mols_list])



## Computational Experiments
![Substructure Pooling Overview](/figures/sub_pool_methods_overview.png)

* The computational experiments from the paper can be reproduced and visualised using the Jupyter notebook [substructure_pooling_experiments.ipynb](substructure_pooling_experiments.ipynb) which provides an easy way to interface with the code base in [modules](modules).
* The [data folder](data) contains the five (cleaned) chemical data sets for the diverse set of prediction tasks investigated in the paper: lipophilicity, aqueous solubility, SARS-CoV-2 main protease inhibition, mutagenicity, and estrogen receptor alpha antagonism. Each data set is given as a set of labelled SMILES strings.
* The computational environment in which the original results were conducted is given in [environment.yml](environment.yml).
* The original numerical results from the paper can be found both in [results](results) and in [results_original](results_original). If new computational results are generated and saved using the Jupyter notebook, then by default they overwrite the content of the [results-folder](results).

