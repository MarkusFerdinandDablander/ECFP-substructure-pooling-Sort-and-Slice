import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


def oversample(X,y, random_seed = 42):
    """
    X.shape = (n_samples, n_features)
    y.shape = (n_samples,)
    y only contains 0s and 1s, indicating the class membership, y can be a list or a numpy array
    """

    X = np.array(X)
    y = np.array(y)

    n_samples = len(y)
    n_positive_samples = np.sum(np.array(y))
    n_negative_samples = n_samples - n_positive_samples
    delta = int(abs(n_positive_samples - n_negative_samples))

    if n_negative_samples <= n_positive_samples:
        minority_class = 0
    else:
        minority_class = 1

    ind_minority_class = np.where(y == minority_class)[0]
    X_minority = X[ind_minority_class]

    np.random.seed(random_seed)
    surplus_ind = np.random.randint(len(X_minority), size = delta)
    X_surplus = X_minority[surplus_ind]

    y_surplus = (np.ones(delta)*minority_class).astype(int)

    X_oversampled = np.append(X, X_surplus, axis = 0)
    y_oversampled = np.append(y, y_surplus)

    return (X_oversampled, y_oversampled)



def undersample(X,y):
    """
    X.shape = (n_samples, n_features)
    y.shape = (n_samples,)
    y only contains 0s and 1s, indicating the class membership, y can be a list of a numpy array
    """

    X = np.array(X)
    y = np.array(y)

    n_samples = len(y)
    n_positive_samples = np.sum(np.array(y))
    n_negative_samples = n_samples - n_positive_samples
    delta = int(abs(n_positive_samples - n_negative_samples))

    if n_negative_samples <= n_positive_samples:
        majority_class = 1
    else:
        majority_class = 0

    ind_majority_class = np.where(y == majority_class)[0]

    np.random.seed(42)
    ind_delete = np.random.choice(ind_majority_class, delta, replace = False)

    X_undersampled = np.delete(X, ind_delete, axis = 0)
    y_undersampled = np.delete(y, ind_delete)

    return (X_undersampled, y_undersampled)



def create_rf_model(ml_settings, task_type, balanced = False):
    
    if task_type == "regression":
        
        rf_model = RandomForestRegressor(n_estimators = ml_settings["n_estimators"],
                                         criterion = ml_settings["criterion"],
                                         max_depth = ml_settings["max_depth"],
                                         min_samples_leaf = ml_settings["min_samples_leaf"],
                                         min_samples_split = ml_settings["min_samples_split"],
                                         max_features = ml_settings["max_features"],
                                         bootstrap = ml_settings["bootstrap"], 
                                         random_state = ml_settings["random_state"])
        
    elif task_type == "classification":
    
        if balanced == False:
        
            rf_model = RandomForestClassifier(n_estimators = ml_settings["n_estimators"],
                                              criterion = ml_settings["criterion"],
                                              max_depth = ml_settings["max_depth"],
                                              min_samples_leaf = ml_settings["min_samples_leaf"],
                                              min_samples_split = ml_settings["min_samples_split"],
                                              max_features = ml_settings["max_features"],
                                              bootstrap = ml_settings["bootstrap"], 
                                              random_state = ml_settings["random_state"])
        else:
    	
            rf_model = BalancedRandomForestClassifier(n_estimators = ml_settings["n_estimators"],
		                                  criterion = ml_settings["criterion"],
		                                  max_depth = ml_settings["max_depth"],
		                                  min_samples_leaf = ml_settings["min_samples_leaf"],
		                                  min_samples_split = ml_settings["min_samples_split"],
		                                  max_features = ml_settings["max_features"],
		                                  bootstrap = ml_settings["bootstrap"], 
		                                  random_state = ml_settings["random_state"])
    	

    return rf_model


   
def make_rf_prediction(rf_model, X_test, task_type):
    
    if task_type == "regression":
        return rf_model.predict(X_test)
    elif task_type == "classification":
        return rf_model.predict_proba(X_test)[:,1]
