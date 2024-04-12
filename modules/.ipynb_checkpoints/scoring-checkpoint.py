import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix, mean_absolute_error, mean_squared_error, median_absolute_error, max_error, r2_score, average_precision_score
from .molecular_featurisation import ecfp_invariants, fcfp_invariants



def transform_probs_to_labels(y_pred_proba_pos, cutoff_value = 0.5):
    """
    Transforms an array of probabilities into a binary array of 0s and 1s.
    """

    y_pred_proba_pos = np.array(y_pred_proba_pos)
    y_pred = np.copy(y_pred_proba_pos)

    y_pred[y_pred > cutoff_value] = 1
    y_pred[y_pred <= cutoff_value] = 0 # per default, sklearn random forest classifiers map a probability of 0.5 to class 0

    return y_pred



def binary_classification_scores(y_true, y_pred_proba_pos, display_results = False):
    """
    For a binary classification task with true labels y_true and predicted probabilities for the positive class y_pred_proba_pos, this function computes the following metrics:
    
    "PRC-AUC",
    "AUROC", 
    "Accuracy", 
    "Balanced Accuracy", 
    "F1-Score", 
    "MCC", 
    "Sensitivity", 
    "Specificity", 
    "Precision", 
    "Negative Predictive Value", 
    "Test Cases", 
    "Negative Test Cases", 
    "Positive Test Cases".
    
    """
    
    if len(y_true) == 0:
        
        # collect scores
        scores_array = np.array([float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), 0, 0, 0])
        scores_array_2d = np.reshape(scores_array, (1, len(scores_array)))
        columns = ["PRC-AUC", "AUROC", "Accuracy", "Balanced Accuracy", "F1-Score", "MCC", "Sensitivity", "Specificity", "Precision", "Negative Predictive Value", "Test Cases", "Negative Test Cases", "Positive Test Cases"]
        scores_df = pd.DataFrame(data = scores_array_2d, index = ["Scores:"], columns = columns)

        # display scores
        if display_results == True:
            display(scores_df)

        return scores_df
        
    else:

        # prepare variables
        y_true = list(y_true)
        y_pred_proba_pos = list(y_pred_proba_pos)
        y_pred = list(transform_probs_to_labels(y_pred_proba_pos))

        n_test_cases = len(y_true)
        n_test_cases_neg = list(y_true).count(0)
        n_test_cases_pos = list(y_true).count(1)

        if y_true.count(y_true[0]) != len(y_true):
            conf_matrix = confusion_matrix(y_true, y_pred)
            tn = conf_matrix[0,0]
            fn = conf_matrix[1,0]
            tp = conf_matrix[1,1]
            fp = conf_matrix[0,1]

        elif y_true.count(0) == len(y_true):

            tn = y_pred.count(0)
            fn = 0
            tp = 0
            fp = y_pred.count(1)

        elif y_true.count(1) == len(y_true):

            tn = 0
            fn = y_pred.count(0)
            tp = y_pred.count(1)
            fp = 0

        # compute scores
        
        # prc_auc
        if y_true.count(y_true[0]) != len(y_true):
            prc_auc = average_precision_score(y_true, y_pred_proba_pos)
        else:
            prc_auc = float("NaN")
        
        # roc_auc
        if y_true.count(y_true[0]) != len(y_true):
            roc_auc = roc_auc_score(y_true, y_pred_proba_pos)
        else:
            roc_auc = float("NaN")

        # accuracy
        accuracy = (tn + tp)/n_test_cases

        # balanced accuracy
        if fn+tp != 0 and fp+tn != 0:
            balanced_accuracy = ((tp/(fn+tp))+(tn/(fp+tn)))/2
        else:
            balanced_accuracy = float("NaN")

        # f1 score
        f1 = f1_score(y_true, y_pred)

        # mcc score
        mcc = matthews_corrcoef(y_true, y_pred)

        # sensitivity
        if fn + tp != 0:
            sensitivity = tp/(fn+tp)
        else:
            sensitivity = float("NaN")

        #specificity
        if fp+tn != 0:
            specificity = tn/(fp+tn)
        else:
            specificity = float("NaN")

        # positive predictive value
        if tp+fp != 0:
            positive_predictive_value = tp/(tp+fp)
        else:
            positive_predictive_value = float("NaN")

        # negative predictive value
        if tn+fn != 0:
            negative_predictive_value = tn/(tn+fn)
        else:
            negative_predictive_value = float("NaN")

        # collect scores
        scores_array = np.array([prc_auc, roc_auc, accuracy, balanced_accuracy, f1, mcc, sensitivity, specificity, positive_predictive_value, negative_predictive_value, n_test_cases, n_test_cases_neg, n_test_cases_pos])
        scores_array_2d = np.reshape(scores_array, (1, len(scores_array)))
        columns = ["PRC-AUC", "AUROC", "Accuracy", "Balanced Accuracy", "F1-Score", "MCC", "Sensitivity", "Specificity", "Precision", "Negative Predictive Value", "Test Cases", "Negative Test Cases", "Positive Test Cases"]
        scores_df = pd.DataFrame(data = scores_array_2d, index = ["Scores:"], columns = columns)

        # display scores
        if display_results == True:
            display(scores_df)

        return scores_df



def regression_scores(y_true, y_pred, display_results = False):
    
    """
    For a regression task with true labels y_true and predicted labels y_pred, this function computes the following metrics:
    
    "MAE", 
    "MedAE", 
    "RMSE", 
    "MaxAE", 
    "MSE", 
    "PearsonCorr", 
    "R2Coeff", 
    "Test Cases".
    
    """
    
    if len(y_true) == 0:
        
        # collect scores
        scores_array = np.array([float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), 0])
        scores_array_2d = np.reshape(scores_array, (1, len(scores_array)))
        columns = ["MAE", "MedAE", "RMSE", "MaxAE", "MSE", "PearsonCorr", "R2Coeff", "Test Cases"]
        scores_df = pd.DataFrame(data = scores_array_2d, index = ["Scores:"], columns = columns)

        # display scores
        if display_results == True:
            display(scores_df)

        return scores_df
        
    else:

        # prepare variables
        y_true = list(y_true)
        y_pred = list(y_pred)
        n_test_cases = len(y_true)

        # compute scores

        # mean absolute error
        mae = mean_absolute_error(y_true, y_pred)

        # median absolute error
        medae = median_absolute_error(y_true, y_pred)

        # root mean squared error
        rmse = mean_squared_error(y_true, y_pred, squared = False)

        # max error
        maxe = max_error(y_true, y_pred)

        # mean squared error
        mse = mean_squared_error(y_true, y_pred, squared = True)

        # pearson correlation coefficient
        pearson_corr = pearsonr(y_true, y_pred)[0]
        
        # R2 coefficient of determination
        r2_coeff = r2_score(y_true, y_pred)

        # collect scores
        scores_array = np.array([mae, medae, rmse, maxe, mse, pearson_corr, r2_coeff, n_test_cases])
        scores_array_2d = np.reshape(scores_array, (1, len(scores_array)))
        columns = ["MAE", "MedAE", "RMSE", "MaxAE", "MSE", "PearsonCorr", "R2Coeff", "Test Cases"]
        scores_df = pd.DataFrame(data = scores_array_2d, index = ["Scores:"], columns = columns)

        # display scores
        if display_results == True:
            display(scores_df)

        return scores_df



def summarise_display_and_save_results_and_settings(scores_dict, settings_dict, display_results = True):
    
    "Record and save performance results in scores_dict after computational experiment with settings_dict has been run."
    
    columns = list(scores_dict[(0,0)].columns)
    scores_table = pd.DataFrame(columns = columns)
    cv_column = []
    
    for (i,j) in scores_dict.keys():
        
        cv_column.append((i,j))
        scores_table = scores_table.append(scores_dict[(i,j)])
        
    scores_table.index = cv_column
    
    scores_table.loc["mean"] = scores_table.mean()
    scores_table.loc["std"] = scores_table.std()
    
    if display_results == True:
        display(scores_table)
    
    if settings_dict["ecfp_settings"]["mol_to_invs_function"] == ecfp_invariants:
        feature_info = str(settings_dict["ecfp_settings"]["dimension"]) + "ecfp" + str(2*settings_dict["ecfp_settings"]["radius"]) + settings_dict["ecfp_settings"]["pool_method"]
    elif settings_dict["ecfp_settings"]["mol_to_invs_function"] == fcfp_invariants:
        feature_info = str(settings_dict["ecfp_settings"]["dimension"]) + "fcfp" + str(2*settings_dict["ecfp_settings"]["radius"]) + settings_dict["ecfp_settings"]["pool_method"]
    
    filename = settings_dict["dataset_abbrev"] + "_" + settings_dict["split_type"] + "k" + str(settings_dict["k_splits"]) + "m" + str(settings_dict["m_reps"]) + "_" + feature_info + "_" + settings_dict["ml_model"]
    filepath = "results/" + settings_dict["dataset_name"] + "/" + settings_dict["split_type"] + "/"
    scores_table.to_csv(filepath + filename + ".csv")
    
    with open(filepath + "settings_dict_last_experiment.txt", 'w') as f:
        for key in settings_dict.keys():
            f.write(str(key) + " = " + str(settings_dict[key]) + "\n")
