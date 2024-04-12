import numpy as np
import pandas as pd
import random
from chembl_structure_pipeline.standardizer import standardize_mol, get_parent_mol
from .utils import discretise, remove_random_element
from .graph_theory import extract_labelled_circular_subgraph_object, check_if_strict_labelled_subgraph
from .information_theory import *
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency



def smiles_standardisation(x_smiles, get_parent_smiles = True):

    standardised_x_smiles = list(range(len(x_smiles)))

    for (k, smiles) in enumerate(x_smiles):

        try:

            # convert smiles to mol object
            mol = Chem.MolFromSmiles(smiles)

            # standardise mol object
            standardised_mol = standardize_mol(mol, check_exclusion = True)

            if get_parent_smiles == True:

                # systematically remove salts, solvents and isotopic information to get parent mol
                (standardised_mol, exclude) = get_parent_mol(standardised_mol,
                                                             neutralize = True,
                                                             check_exclusion=True,
                                                             verbose = False)

            # convert mol object back to smiles
            standardised_smiles = Chem.MolToSmiles(standardised_mol)

            # replace smiles with standardised parent smiles
            standardised_x_smiles[k] = standardised_smiles

        except:

            # leave smiles unchanged if it generates an exception
            standardised_x_smiles[k] = smiles

    return np.array(standardised_x_smiles)



def random_invariants(mol):
    
    invs = random.sample(range(1, 1000000), mol.GetNumAtoms())
    
    return invs



def uniform_invariants(mol):
    
    invs = [1]*mol.GetNumAtoms()
    
    return invs



def atomic_number_invariants(mol):
    
    invs = []
    
    for atom in mol.GetAtoms():
        invs.append(atom.GetAtomicNum())
    
    return invs



def ecfp_invariants(mol):
    
    invs = []
    
    for k in range(mol.GetNumAtoms()):
    
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius = 0, atomInvariantsGenerator = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership = True))
        fingerprint = fp_gen.GetSparseCountFingerprint(mol, fromAtoms = [k])        
        invs.append(list(fingerprint.GetNonzeroElements().keys())[0])

    return invs



def fcfp_invariants(mol):
    
    invs = []
    
    for k in range(mol.GetNumAtoms()):
    
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius = 0, atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
        fingerprint = fp_gen.GetSparseCountFingerprint(mol, fromAtoms = [k])        
        invs.append(list(fingerprint.GetNonzeroElements().keys())[0])

    return invs



def combined_invariants(invariant_func_1, invariant_func_2):
    
    def combined_invariants(mol):
        
        invs_1 = invariant_func_1(mol)
        invs_2 = invariant_func_2(mol)
        invs = [hash((i_1, i_2)) % 10000001 for (i_1, i_2) in list(zip(invs_1, invs_2))]
        
        return invs
    
    return combined_invariants



def ecfp_atom_ids_from_smiles(smiles, ecfp_settings):
    
    mol = Chem.MolFromSmiles(smiles)
    
    info_dict = {}
    
    fp = GetMorganFingerprint(mol,
                              radius = ecfp_settings["radius"],
                              useCounts = ecfp_settings["use_counts"],
                              invariants = ecfp_settings["mol_to_invs_function"](mol), 
                              useChirality = ecfp_settings["use_chirality"], 
                              useBondTypes = ecfp_settings["use_bond_invs"], 
                              bitInfo = info_dict)

    return (UIntSparseIntVect.GetNonzeroElements(fp), info_dict)



def one_hot_vec(dim, k):
    
    vec = np.zeros(dim)
    vec[k] = 1
    
    return vec.astype(int)



def create_ecfp_atom_id_one_hot_encoder_frequency(x_smiles, ecfp_settings):
    
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
    
    atom_id_to_support_cardinality = {atom_id: sum(support_list) for (atom_id, support_list) in atom_id_to_support_list.items()}
    atom_id_list_sorted = sorted(list(atom_id_set), key = lambda atom_id: atom_id_to_support_cardinality[atom_id], reverse = True)
    
    final_atom_id_list = atom_id_list_sorted[0: ecfp_settings["dimension"]]

    zero_padding_dim = int(-min(len(final_atom_id_list) - ecfp_settings["dimension"], 0)) + 1
    final_atom_id_list_to_one_hot_vecs = dict([(atom_id, one_hot_vec(len(final_atom_id_list) + zero_padding_dim, k)) for (k, atom_id) in enumerate(final_atom_id_list)])
    
    def atom_id_one_hot_encoder(atom_id):
        
        other_vec = one_hot_vec(len(final_atom_id_list) + zero_padding_dim, len(final_atom_id_list) + zero_padding_dim - 1)
        
        return final_atom_id_list_to_one_hot_vecs.get(atom_id, other_vec)
    
    return atom_id_one_hot_encoder



def chi_squared_p_value(x_discrete, y_discrete):
    
    df = pd.DataFrame(data = {"x_discrete": x_discrete, "y_discrete": y_discrete})
    contingency_table = pd.crosstab(index = df["x_discrete"], columns = df["y_discrete"]).values
    (chi_squared, p_value, degrees_of_freedom, expected_frequencies) = chi2_contingency(contingency_table)

    return p_value



def create_ecfp_atom_id_one_hot_encoder_chi_squared(x_smiles, y, ecfp_settings, discretise_y, random_state = 42):
    
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
    
    zero_padding_dim = int(-min(len(final_atom_id_list) - ecfp_settings["dimension"], 0)) + 1
    final_atom_id_list_to_one_hot_vecs = dict([(atom_id, one_hot_vec(len(final_atom_id_list) + zero_padding_dim, k)) for (k, atom_id) in enumerate(final_atom_id_list)])
    
    def atom_id_one_hot_encoder(atom_id):
        
        other_vec = one_hot_vec(len(final_atom_id_list) + zero_padding_dim, len(final_atom_id_list) + zero_padding_dim - 1)
        
        return final_atom_id_list_to_one_hot_vecs.get(atom_id, other_vec)
    
    return atom_id_one_hot_encoder



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
    
    final_atom_id_list = atom_id_list_sorted[0: ecfp_settings["dimension"]]

    zero_padding_dim = int(-min(len(final_atom_id_list) - ecfp_settings["dimension"], 0)) + 1
    final_atom_id_list_to_one_hot_vecs = dict([(atom_id, one_hot_vec(len(final_atom_id_list) + zero_padding_dim, k)) for (k, atom_id) in enumerate(final_atom_id_list)])
    
    def atom_id_one_hot_encoder(atom_id):
        
        other_vec = one_hot_vec(len(final_atom_id_list) + zero_padding_dim, len(final_atom_id_list) + zero_padding_dim - 1)
        
        return final_atom_id_list_to_one_hot_vecs.get(atom_id, other_vec)
    
    return atom_id_one_hot_encoder



def create_ecfp_atom_id_one_hot_encoder_cmim(x_smiles, y, ecfp_settings, discretise_y, entropy_keep_upper = 8192, base = 2):
    
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
    
    # step 1: sort fragments by entropy and then only keep entropy_keep_upper 
    atom_ids_sorted_by_entropy = sorted(list(atom_id_set), key = lambda atom_id: entropy(atom_id_to_support_list[atom_id], base = base), reverse = False)
    
    while len(atom_id_set) > max(ecfp_settings["dimension"], entropy_keep_upper):
        
        atom_id_set.remove(atom_ids_sorted_by_entropy[0])
        atom_id_to_support_list.pop(atom_ids_sorted_by_entropy[0])
        atom_id_to_support_list_with_counts.pop(atom_ids_sorted_by_entropy[0])
        atom_ids_sorted_by_entropy.remove(atom_ids_sorted_by_entropy[0])

    # step 2: select fragments via conditional mutual information maximisation
    if discretise_y == True:
        y = discretise(y, n_bins = 2, strategy = "quantile")

    final_atom_id_list = fast_feature_selection_via_cmi_maximisation(atom_id_to_support_list, y, ecfp_settings["dimension"], base = base)

    zero_padding_dim = int(-min(len(final_atom_id_list) - ecfp_settings["dimension"], 0)) + 1
    final_atom_id_list_to_one_hot_vecs = dict([(atom_id, one_hot_vec(len(final_atom_id_list) + zero_padding_dim, k)) for (k, atom_id) in enumerate(final_atom_id_list)])
    
    def atom_id_one_hot_encoder(atom_id):
        
        other_vec = one_hot_vec(len(final_atom_id_list) + zero_padding_dim, len(final_atom_id_list) + zero_padding_dim - 1)
        
        return final_atom_id_list_to_one_hot_vecs.get(atom_id, other_vec)
    
    return atom_id_one_hot_encoder



def create_ecfp_vector_multiset(smiles, ecfp_settings, atom_id_one_hot_encoder):
    
    atom_id_dict = ecfp_atom_ids_from_smiles(smiles, ecfp_settings)[0]
    atom_id_list = []
    
    for key in atom_id_dict:
        atom_id_list += [key]*atom_id_dict[key]
    
    vector_multiset = np.array([atom_id_one_hot_encoder(atom_id) for atom_id in atom_id_list])
    vector_multiset = np.delete(vector_multiset, np.where(vector_multiset[:,-1] == 1)[0], axis = 0)[:,0:-1]
    
    return vector_multiset



def create_ecfp_featuriser(ecfp_settings,
                           x_smiles_train = None,
                           y_train = None,
                           discretise_y = None,
                           base = 2,
                           random_state = 42):
    
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
        
        if ecfp_settings["pool_method"] == "sorted":
            atom_id_one_hot_encoder = create_ecfp_atom_id_one_hot_encoder_frequency(x_smiles_train, ecfp_settings)
        elif ecfp_settings["pool_method"] == "chi2":
            atom_id_one_hot_encoder = create_ecfp_atom_id_one_hot_encoder_chi_squared(x_smiles_train, y_train, ecfp_settings, discretise_y, random_state)
        elif ecfp_settings["pool_method"] == "mim":
            atom_id_one_hot_encoder = create_ecfp_atom_id_one_hot_encoder_mim(x_smiles_train, y_train, ecfp_settings, discretise_y, base, random_state)
        
        def featuriser(x_smiles):
            
            X_fp = np.zeros((len(x_smiles), ecfp_settings["dimension"]))

            for (k, smiles) in enumerate(x_smiles):
                
                X_fp[k,:] = np.sum(create_ecfp_vector_multiset(smiles, ecfp_settings, atom_id_one_hot_encoder), axis = 0)
                
            return X_fp
        
        return featuriser






