import math
from collections import defaultdict



def p_log_p(p, base = 2):
    
    assert 0 <= p and p <= 1
    
    if 0 < p and p < 1:
        return p*math.log(p, base)
    
    else:
        return 0



def create_prob_list(list_hash):
    """
    Gives a list of probabilities [p_1, ..., p_n] with p_1 + ... + p_n = 1 for a list of hashable objects based on the relative frequencies of the objects.
    """

    entities_to_counts = defaultdict(int)
    
    for entity in list_hash:
        entities_to_counts[entity] += 1
    
    len_list_hash = sum(entities_to_counts.values())
        
    prob_list = (count/len_list_hash for count in entities_to_counts.values())

    return prob_list



def entropy_from_prob_list(prob_list, base = 2):
    """
    Gives the entropy H for a list of probabilities [p_1,...,p_n] such that p_1 + ... + p_n = 1.
    """

    return -sum(p_log_p(p, base) for p in prob_list)



def entropy(list_hash, base = 2):
    """ 
    Gives the entropy H for a list of hashable objects based on the relative frequencies of the objects
    """
    return entropy_from_prob_list(create_prob_list(list_hash), base)



def mi(list_hash_1, list_hash_2, base = 2):
    """
    Gives the mutual information I of two lists of hashable objects X and Y based on the identity I(X,Y) = H(X) + H(Y) - H(X,Y).
    """
    
    return entropy(list_hash_1, base) + entropy(list_hash_2, base) - entropy(zip(list_hash_1, list_hash_2), base)