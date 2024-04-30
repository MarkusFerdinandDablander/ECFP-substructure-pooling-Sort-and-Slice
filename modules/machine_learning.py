from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



def create_rf_model(ml_settings, task_type):
    
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
    
        rf_model = RandomForestClassifier(n_estimators = ml_settings["n_estimators"],
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
        # predict continuous labels
        return rf_model.predict(X_test)
    elif task_type == "classification":
        # predict probabilties for positive class
        return rf_model.predict_proba(X_test)[:,1]
