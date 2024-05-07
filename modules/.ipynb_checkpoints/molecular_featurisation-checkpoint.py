import numpy as np
import pandas as pd
import random
from collections import defaultdict
from .graph_theory import extract_labelled_circular_subgraph_object, check_if_strict_labelled_subgraph
from .information_theory import mi
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency



def ecfp_invariants(mol):
    """
    A function that maps an RDKit mol object to a list of hashed integer identifiers (= initial atom ids) describing the initial standard ECFP atomic invariants.
    """
    
    invs = []
    
    for k in range(mol.GetNumAtoms()):
    
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius = 0, atomInvariantsGenerator = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership = True))
        fingerprint = fp_gen.GetSparseCountFingerprint(mol, fromAtoms = [k])        
        invs.append(list(fingerprint.GetNonzeroElements().keys())[0])

    return invs



def fcfp_invariants(mol):
    """
     A function that maps an RDKit mol object to a list of hashed integer identifiers (= initial atom ids) describing the initial pharmacophoric FCFP atomic invariants.
    """
    
    invs = []
    
    for k in range(mol.GetNumAtoms()):
    
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius = 0, atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
        fingerprint = fp_gen.GetSparseCountFingerprint(mol, fromAtoms = [k])        
        invs.append(list(fingerprint.GetNonzeroElements().keys())[0])

    return invs



def one_hot_vec(dim, k):
    """
    Creates a one-hot vector that has 0s everywhere except in its k-th component.
    """
    
    vec = np.zeros(dim)
    vec[k] = 1
    
    return vec.astype(int)



def ecfp_atom_ids_from_smiles(smiles, ecfp_settings):
    """
    A function that takes as input a SMILES string and a dictionary of ECFP settings and outputs a pair of dictionaries. The keys of each dictionary are given by the hashed integer ECFP substructure identifiers ( = the set of generated atom ids) in the input molecule. The first dictionary maps each atom id to its count (i.e. how often it appears in the input compound). If ecfp_settings["use_counts"] = False, then all atom ids are mapped to 1. The second dictionary named info_dict maps each atom id to a tuple containing the location(s) of the associated circular substructure in the input compound (specified via the center atom and the radius).
    
    """
    
    mol = Chem.MolFromSmiles(smiles)
    
    info_dict = {}
    
    fp = GetMorganFingerprint(mol,
                              radius = ecfp_settings["radius"], # 0 or 1 or 2 or ...
                              useCounts = ecfp_settings["use_counts"], # True or False
                              invariants = ecfp_settings["mol_to_invs_function"](mol), # ecfp_invariants(mol) or fcfp_invariants(mol) 
                              useChirality = ecfp_settings["use_chirality"], # True or False
                              useBondTypes = ecfp_settings["use_bond_invs"], # True or False
                              bitInfo = info_dict)

    return (UIntSparseIntVect.GetNonzeroElements(fp), info_dict)



def create_ecfp_atom_id_one_hot_encoder_sort_and_slice(x_smiles, ecfp_settings, break_ties_with = "sorted_list_command"):
    """
    Takes as input a list of SMILES strings x_smiles = [smiles_1, smiles_2, ...] and a dictionary of ECFP settings and gives as output a function atom_id_one_hot_encoder that maps 
    integer substructure identifiers (= atom ids) to one-hot encoded vector representations of dimension ecfp_settings["dimension"]. 
    The components of the vector representation are sorted by prevalence, whereby the substructure identifiers that appear in the most compounds in x_smiles appear first.
    Ties are either broken using the arbitrary ordering induced via the command sorted(list(...), ...) or via the abritrary ordering induced by the integer atom ids themselves.
    Substructure identifiers that are not part of the most frequent ecfp_settings["dimension"] substructures (or not in x_smiles at all) are all mapped to a vector of 0s. 
    If ecfp_settings["dimension"] is larger than the number of unique substructure identifiers in all compounds in x_smiles, then the vector representation is padded with 0s 
    to reach the desired length. 
    """
    
    # preallocate data structures
    atom_id_set = set()
    atom_id_to_support_list = defaultdict(lambda: [0]*len(x_smiles)) # the support_list for a given atom_id is a binary list of len(x_smiles) whose i-th entry specifies whether the atom_id is contained in smiles_i
    
    # create set of all atom id in x_smiles and binary substructural training feature vector for each atom_id in the form of a dictionary atom_id_to_support_list
    for (k, smiles) in enumerate(x_smiles):
        
        current_atom_id_to_count = ecfp_atom_ids_from_smiles(smiles, ecfp_settings)[0]
        atom_id_set = atom_id_set.union(set(current_atom_id_to_count))
        
        for atom_id in current_atom_id_to_count.keys():
            atom_id_to_support_list[atom_id][k] = 1 if current_atom_id_to_count[atom_id] > 0 else 0
            
    # create list of atom ids sorted by prevalence in x_smiles (ties are either broken using the arbitrary ordering induced via sorted(list(...), ...) or the arbitrary ordering induced by the integer atom ids themselves)
    if break_ties_with == "sorted_list_command":
        atom_id_list_sorted = sorted(list(atom_id_set), key = lambda atom_id: sum(atom_id_to_support_list[atom_id]), reverse = True)
    elif break_ties_with == "atom_ids":
        atom_id_list_sorted = sorted(list(atom_id_set), key = lambda atom_id: (sum(atom_id_to_support_list[atom_id]), -atom_id), reverse = True)
    
    # create integer substructure identifier (= atom id) embedding function
    def atom_id_one_hot_encoder(atom_id):
        
        return one_hot_vec(ecfp_settings["dimension"], atom_id_list_sorted.index(atom_id)) if atom_id in atom_id_list_sorted[0: ecfp_settings["dimension"]] else np.zeros(ecfp_settings["dimension"])
    
    print("Number of unique substructures in molecular data set = ", len(atom_id_to_support_list.keys()))
    
    return atom_id_one_hot_encoder



def discretise(y_cont, n_bins = 2, strategy = "uniform"):
    """
    Discretise continuous array.
    """
    
    discretiser = KBinsDiscretizer(n_bins = n_bins, encode = "ordinal", strategy = strategy)
    y_disc = list(discretiser.fit_transform(np.array(y_cont).reshape(-1,1)).reshape(-1).astype(int))
    
    return y_disc



def chi_squared_p_value(x_discrete, y_discrete):
    """
    Return p-value of a Chi-squared significance test between two samples x_discrete and y_discrete.
    """
    
    df = pd.DataFrame(data = {"x_discrete": x_discrete, "y_discrete": y_discrete})
    contingency_table = pd.crosstab(index = df["x_discrete"], columns = df["y_discrete"]).values
    (chi_squared, p_value, degrees_of_freedom, expected_frequencies) = chi2_contingency(contingency_table)

    return p_value



def create_ecfp_atom_id_one_hot_encoder_filtered(x_smiles, y, ecfp_settings, discretise_y, random_state = 42):
    
    # set random seed
    random.seed(random_state)
    
    # preallocate data structures
    atom_id_set = set()
    atom_id_to_support_list_with_counts = {}
    atom_id_to_info_list = {}
    
    # create set of all occuring atom ids and associated feature matrix with support columns
    for (k, smiles) in enumerate(x_smiles):
        
        (current_atom_id_to_count, current_info) = ecfp_atom_ids_from_smiles(smiles, ecfp_settings)
        atom_id_set = atom_id_set.union(set(current_atom_id_to_count))
        
        for atom_id in set(current_atom_id_to_count):
            atom_id_to_support_list_with_counts[atom_id] = atom_id_to_support_list_with_counts.get(atom_id, [0]*len(x_smiles))
            atom_id_to_support_list_with_counts[atom_id][k] = current_atom_id_to_count[atom_id]
            atom_id_to_info_list[atom_id] = atom_id_to_info_list.get(atom_id, []) + [(smiles, current_info[atom_id])]
    
    # binarise support list so that it only indicates presence/absence of fragments in training compounds
    atom_id_to_support_list = {atom_id: [1 if b > 0 else 0 for b in support_list_with_counts] for (atom_id, support_list_with_counts) in atom_id_to_support_list_with_counts.items()}
    
    print("Number of unique substructures = ", len(atom_id_set))
    
    # step 1: remove fragments with support of cardinality = 1, i.e. fragments which only occur in a single training compound
    atom_ids_single_occ = [atom_id for atom_id in atom_id_set if sum(atom_id_to_support_list[atom_id]) == 1]
    random.shuffle(atom_ids_single_occ)

    while len(atom_ids_single_occ) > 0 and len(atom_id_set) > ecfp_settings["dimension"]:
        
        atom_id_set.remove(atom_ids_single_occ[0])
        atom_id_to_support_list.pop(atom_ids_single_occ[0])
        atom_id_to_support_list_with_counts.pop(atom_ids_single_occ[0])
        atom_ids_single_occ.remove(atom_ids_single_occ[0])
    
    # step 2: remove non-closed fragments, i.e. fragments for which a subfragment (an labelled subgraph) exists which occurs in the exact same set of training compounds (= which has the same support)
    support_tuple_to_atom_id_list = {}
    for (atom_id, support_list) in atom_id_to_support_list.items():
        support_tuple_to_atom_id_list[tuple(support_list)] = support_tuple_to_atom_id_list.get(tuple(support_list), []) + [atom_id]
    
    atom_id_lists_grouped_by_support = [atom_id_list for atom_id_list in support_tuple_to_atom_id_list.values() if len(atom_id_list) >= 2]
    atom_id_to_graph_object = {}
    
    for atom_id_list in atom_id_lists_grouped_by_support:
        for atom_id in atom_id_list:
            
            (smiles, positions) = random.choice(atom_id_to_info_list[atom_id])
            (center_atom_index, radius) = random.choice(positions)
            atom_id_to_graph_object[atom_id] = extract_labelled_circular_subgraph_object(Chem.MolFromSmiles(smiles), center_atom_index, radius, ecfp_settings)
            
    atom_ids_non_closed = []
    
    for atom_id_list in atom_id_lists_grouped_by_support:
        for atom_id in atom_id_list:
            if sum([check_if_strict_labelled_subgraph(atom_id_to_graph_object[other_atom_id], atom_id_to_graph_object[atom_id]) for other_atom_id in atom_id_list]) > 0:
                atom_ids_non_closed.append(atom_id)
            
    random.shuffle(atom_ids_non_closed)

    while len(atom_ids_non_closed) > 0 and len(atom_id_set) > ecfp_settings["dimension"]:
        
        atom_id_set.remove(atom_ids_non_closed[0])
        atom_id_to_support_list.pop(atom_ids_non_closed[0])
        atom_id_to_support_list_with_counts.pop(atom_ids_non_closed[0])
        atom_ids_non_closed.remove(atom_ids_non_closed[0])
    
    # step 3: rank fragments via Chi-square test and make cutoff
    if discretise_y == True:
        y = discretise(y, n_bins = 2, strategy = "quantile")
    
    atom_id_list_chi_squared_sorted = sorted(list(atom_id_set), key = lambda atom_id: chi_squared_p_value(atom_id_to_support_list[atom_id], y)) # replace with atom_id_to_support_list_with_counts to include count-information (if switched on in ecfp_settings) when computing the p-values
    
    while len(atom_id_set) > ecfp_settings["dimension"]:
        
        atom_id_set.remove(atom_id_list_chi_squared_sorted[-1])
        atom_id_to_support_list.pop(atom_id_list_chi_squared_sorted[-1])
        atom_id_to_support_list_with_counts.pop(atom_id_list_chi_squared_sorted[-1])
        atom_id_list_chi_squared_sorted.remove(atom_id_list_chi_squared_sorted[-1])

    final_atom_id_list = list(atom_id_set)
    
    # create integer substructure identifier (= atom id) embedding function
    def atom_id_one_hot_encoder(atom_id):
        
        return one_hot_vec(ecfp_settings["dimension"], final_atom_id_list.index(atom_id)) if atom_id in final_atom_id_list[0: ecfp_settings["dimension"]] else np.zeros(ecfp_settings["dimension"])
    
    return atom_id_one_hot_encoder




def remove_random_element(my_list, random_state = 42):
    """
    Remove random element from a list and return the new list.
    """
    
    random.seed(random_state)
    my_list.pop(random.randrange(len(my_list)))
    
    return my_list


def create_ecfp_atom_id_one_hot_encoder_mim(x_smiles, y, ecfp_settings, discretise_y, base = 2, random_state = 42):
    
    # set random seed
    random.seed(random_state)
    
     # preallocate data structures
    atom_id_set = set()
    atom_id_to_support_list_with_counts = {}
    
    # create set of all occuring atom ids and associated feature matrix with support columns
    for (k, smiles) in enumerate(x_smiles):
        
        (current_atom_id_to_count, current_info) = ecfp_atom_ids_from_smiles(smiles, ecfp_settings)
        atom_id_set = atom_id_set.union(set(current_atom_id_to_count))
        
        for atom_id in set(current_atom_id_to_count):
            atom_id_to_support_list_with_counts[atom_id] = atom_id_to_support_list_with_counts.get(atom_id, [0]*len(x_smiles))
            atom_id_to_support_list_with_counts[atom_id][k] = current_atom_id_to_count[atom_id]
            
    # binarise support list so that it only indicates presence/absence of fragments in training compounds
    atom_id_to_support_list = {atom_id: [1 if b > 0 else 0 for b in support_list_with_counts] for (atom_id, support_list_with_counts) in atom_id_to_support_list_with_counts.items()}
    
    print("Number of unique substructures = ", len(atom_id_set))
    
    # step 1: randomly drop fragments for which a fragment with same support exists
    support_tuple_to_atom_id_list = {}
    for (atom_id, support_list) in atom_id_to_support_list.items():
        support_tuple_to_atom_id_list[tuple(support_list)] = support_tuple_to_atom_id_list.get(tuple(support_list), []) + [atom_id]
    
    atom_id_lists_grouped_by_support = [atom_id_list for atom_id_list in support_tuple_to_atom_id_list.values() if len(atom_id_list) >= 2]
    
    atom_ids_redundant = []
    for atom_id_list in atom_id_lists_grouped_by_support:
        atom_ids_redundant += remove_random_element(atom_id_list, random_state = random_state)

    random.shuffle(atom_ids_redundant)

    while len(atom_ids_redundant) > 0 and len(atom_id_set) > ecfp_settings["dimension"]:
        
        atom_id_set.remove(atom_ids_redundant[0])
        atom_id_to_support_list.pop(atom_ids_redundant[0])
        atom_id_to_support_list_with_counts.pop(atom_ids_redundant[0])
        atom_ids_redundant.remove(atom_ids_redundant[0])
    
    print("Number of unique substructures after removal of support duplicates = ", len(atom_id_set))
    
    # step 2: sort fragments by mutual information with (discretised) target y and choose the informative ones
    if discretise_y == True:
        y = discretise(y, n_bins = 2, strategy = "quantile")

    atom_id_to_mi = {atom_id: mi(support_list, y, base = base) for (atom_id, support_list) in atom_id_to_support_list.items()}
    atom_id_list_sorted = sorted(list(atom_id_set), key = lambda atom_id: atom_id_to_mi[atom_id], reverse = True)
    
    # create integer substructure identifier (= atom id) embedding function
    def atom_id_one_hot_encoder(atom_id):
        
        return one_hot_vec(ecfp_settings["dimension"], atom_id_list_sorted.index(atom_id)) if atom_id in atom_id_list_sorted[0: ecfp_settings["dimension"]] else np.zeros(ecfp_settings["dimension"])
    
    return atom_id_one_hot_encoder


    



def create_ecfp_vector_multiset(smiles, ecfp_settings, atom_id_one_hot_encoder):
    """
    Transforms an input SMILES string into a (multi)set of one-hot-encoded substructure embeddings (represented as a 2D numpy array) using ecfp_settings to first transform the compound into a set of integer ECFP substructure identifiers (= atom ids), and then using atom_id_one_hot_encoder to vectorise the atom ids. To remain consistent, the embedding function atom_id_one_hot_encoder should be constructued with the same ecfp_settings as is used in this function.
    """
    
    current_atom_id_to_count = ecfp_atom_ids_from_smiles(smiles, ecfp_settings)[0]
    atom_id_list = []
    
    # create list of integer substructure identifiers (= atom ids) in SMILES string (multiplied by their frequencies if ecfp_settings["use_counts"] = True)
    for (atom_id, count) in current_atom_id_to_count.items():
        atom_id_list += [atom_id]*count
    
    # create a representation of the input compound as a (multi)set of substructure emebeddings
    vector_multiset = np.array([atom_id_one_hot_encoder(atom_id) for atom_id in atom_id_list])
    
    return vector_multiset



def create_ecfp_featuriser(ecfp_settings,
                           x_smiles_train = None,
                           y_train = None,
                           discretise_y = None,
                           base = 2,
                           random_state = 42):
    
    """
    Create a featurisation function "featuriser" that takes as input a list of SMILES strings x_smiles and gives as output a numpy array whose rows are vectorial ECFP representations for the input compounds. The option ecfp_settings["pool_method"] determines which substructure pooling method the output featuriser is equipped with.
    """
    
    if ecfp_settings["pool_method"] == "hashed":
        
        def featuriser(x_smiles):
            
            x_mol = [Chem.MolFromSmiles(smiles) for smiles in x_smiles]
            
            X_fp = np.array([Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                                                 radius = ecfp_settings["radius"],
                                                                                 nBits = ecfp_settings["dimension"],
                                                                                 invariants = ecfp_settings["mol_to_invs_function"](mol),
                                                                                 useBondTypes = ecfp_settings["use_bond_invs"],
                                                                                 useChirality = ecfp_settings["use_chirality"]) for mol in x_mol])
            return X_fp
        
        return featuriser
    
    elif ecfp_settings["pool_method"] != "hashed":
        
        if ecfp_settings["pool_method"] == "sort_and_slice":
            atom_id_one_hot_encoder = create_ecfp_atom_id_one_hot_encoder_sort_and_slice(x_smiles_train, ecfp_settings)
        elif ecfp_settings["pool_method"] == "filtered":
            atom_id_one_hot_encoder = create_ecfp_atom_id_one_hot_encoder_filtered(x_smiles_train, y_train, ecfp_settings, discretise_y, random_state)
        elif ecfp_settings["pool_method"] == "mim":
            atom_id_one_hot_encoder = create_ecfp_atom_id_one_hot_encoder_mim(x_smiles_train, y_train, ecfp_settings, discretise_y, base, random_state)
        
        def featuriser(x_smiles):
            
            X_fp = np.zeros((len(x_smiles), ecfp_settings["dimension"]))

            for (k, smiles) in enumerate(x_smiles):
                
                X_fp[k,:] = np.sum(create_ecfp_vector_multiset(smiles, ecfp_settings, atom_id_one_hot_encoder), axis = 0)
                
            return X_fp
        
        return featuriser







    
    
    
    
    
    
    
    
"""
def create_ecfp_atom_id_one_hot_encoder_sort_and_slice_old(x_smiles, ecfp_settings):
    
     # preallocate data structures
    atom_id_set = set()
    atom_id_to_support_list_with_counts = {}
    
    # create set of all occuring integer substructure identifiers (= atom ids) and associated feature matrix with support columns
    for (k, smiles) in enumerate(x_smiles):
        
        (current_atom_id_to_count, current_info) = ecfp_atom_ids_from_smiles(smiles, ecfp_settings)
        atom_id_set = atom_id_set.union(set(current_atom_id_to_count))
        
        for atom_id in set(current_atom_id_to_count):
            atom_id_to_support_list_with_counts[atom_id] = atom_id_to_support_list_with_counts.get(atom_id, [0]*len(x_smiles))
            atom_id_to_support_list_with_counts[atom_id][k] = current_atom_id_to_count[atom_id]
    
    # binarise support list so that it only indicates presence/absence of fragments in training compounds
    atom_id_to_support_list = {atom_id: [1 if b > 0 else 0 for b in support_list_with_counts] for (atom_id, support_list_with_counts) in atom_id_to_support_list_with_counts.items()}
    
    # create atom id list sorted by prevalence
    atom_id_to_support_cardinality = {atom_id: sum(support_list) for (atom_id, support_list) in atom_id_to_support_list.items()}
    atom_id_list_sorted = sorted(list(atom_id_set), key = lambda atom_id: atom_id_to_support_cardinality[atom_id], reverse = True)
   
    # create integer substructure identifier (= atom id) embedding function
    def atom_id_one_hot_encoder(atom_id):
        
        return one_hot_vec(ecfp_settings["dimension"], atom_id_list_sorted.index(atom_id)) if atom_id in atom_id_list_sorted[0:ecfp_settings["dimension"]] else np.zeros(ecfp_settings["dimension"])

    print("Number of unique substructures in molecular data set = ", len(atom_id_to_support_list.keys()))

    return atom_id_one_hot_encoder

"""