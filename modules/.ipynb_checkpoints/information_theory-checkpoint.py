import math
from collections import defaultdict



def p_log_p(p, base = 2):
    
    assert 0 <= p and p <= 1
    
    if 0 < p and p < 1:
        return p*math.log(p, base)
    
    else:
        return 0



def create_prob_list(list_hash):

    entities_to_counts = defaultdict(int)
    
    for entity in list_hash:
        entities_to_counts[entity] += 1
    
    len_list_hash = sum(entities_to_counts.values())
        
    prob_list = (count/len_list_hash for count in entities_to_counts.values())

    return prob_list



def entropy_from_prob_list(prob_list, base = 2):

    return -sum(p_log_p(p, base) for p in prob_list)



def entropy(list_hash, base = 2):
    return entropy_from_prob_list(create_prob_list(list_hash), base)



def mi(list_hash_1, list_hash_2, base = 2):
    
    return entropy(list_hash_1, base) + entropy(list_hash_2, base) - entropy(zip(list_hash_1, list_hash_2), base)



def cmi(list_hash_1, list_hash_2, list_hash_3, base = 2):
    
    ent_3 = entropy(list_hash_3, base)
    ent_13 = entropy(zip(list_hash_1, list_hash_3), base)
    ent_23 = entropy(zip(list_hash_2, list_hash_3), base)
    ent_123 = entropy(zip(list_hash_1, list_hash_2, list_hash_3), base)

    return ent_13 + ent_23 - ent_3 - ent_123



def feature_selection_via_cmi_maximisation(feature_names_to_vecs, y, n_features, base = 2):
    
    """
    Inputs:
    
    feature_names_to_vecs... dictionary that maps feature names to discrete feature vectors of shape (n_samples,)
    y_discrete... discrete label vector with shape (n_samples,)
    
    Outputs:
    
    chosen_features... list of names of chosen features, ordered via cmim algorithm, with feature maximising mi with y coming first, 
    of shape (n_features,)
    """

    if n_features < len(feature_names_to_vecs.keys()):
    
        feature_names_to_scores = {feature_name : mi(vec, y, base = base) for (feature_name, vec) in feature_names_to_vecs.items()}
        chosen_features = ["_"] * n_features

        for k in range(n_features):

            chosen_features[k] =  max(feature_names_to_scores, key = feature_names_to_scores.get)

            for (feature_name, vec) in feature_names_to_vecs.items():

                current_cmi_value = cmi(vec, y, feature_names_to_vecs[chosen_features[k]], base = base)
                feature_names_to_scores[feature_name] = min(feature_names_to_scores[feature_name], current_cmi_value)
    else:
        
        chosen_features = feature_names_to_vecs.keys()
    
    return chosen_features



def fast_feature_selection_via_cmi_maximisation(feature_names_to_vecs, y, n_features, base = 2):
    """
    Inputs:
    
    feature_names_to_vecs... dictionary that maps feature names to discrete feature vectors of shape (n_samples,)
    y_discrete... discrete label vector with shape (n_samples,)
    
    Outputs:
    
    chosen_features... list of names of chosen features, ordered via cmim algorithm, with feature maximising mi with y coming first, 
    of shape (n_features,)
    """

    if n_features < len(feature_names_to_vecs.keys()):
        
        feature_names_to_scores = {feature_name : mi(vec, y, base = base) for (feature_name, vec) in feature_names_to_vecs.items()}
        feature_names_to_step_count = {feature_name : 0 for feature_name in feature_names_to_vecs.keys()}
        chosen_features = ["_"] * n_features

        for k in range(n_features):

            crit_score = 0

            for (feature_name, vec) in feature_names_to_vecs.items():

                while feature_names_to_scores[feature_name] > crit_score and feature_names_to_step_count[feature_name] < k:

                    current_cmi_value = cmi(vec, y, feature_names_to_vecs[chosen_features[feature_names_to_step_count[feature_name]]], base = base)
                    feature_names_to_scores[feature_name] = min(feature_names_to_scores[feature_name], current_cmi_value)
                    feature_names_to_step_count[feature_name] += 1

                if feature_names_to_scores[feature_name] > crit_score:

                    crit_score = feature_names_to_scores[feature_name]
                    chosen_features[k] = feature_name
    
    else:
        
        chosen_features = feature_names_to_vecs.keys()
        
    return chosen_features