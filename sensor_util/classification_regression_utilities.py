import numpy as np
import pandas as pd
import statistics
from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score, roc_auc_score, mean_absolute_error, r2_score
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from .feature_selection import greedily_remove_features_by_correlation_with_each_other, greedily_remove_features_by_correlation_with_target


def print_classification_report_binary(
    true_labels, predicted_labels, predicted_scores
):
    """
    Prints accuracy, auc, precision, sensivity, specificity, and confusion matrix, given true and predicted binary labels and scores
    
    Parameters
    ----------
    true_labels: array-like
        Assumed to be a binary classification (0 or 1) - if not, all the classifications besides 1 will be grouped together as 
        the negative condition
    predicted_labels: array-like 
        Assumed to be a binary classification (0 or 1) - if not, all the classifications besides 1 will be grouped together as 
        the negative condition
    predicted_scores: array-like
        Scores between 0 and one giving classification score (e.g. probability that condition=1, accroding to the model)
    
    Returns
    -------
    None

    Notes
    -----
    sklearn.metrics.classification report is an alternative
    """

    s = calculate_sensitivity_and_specificity(true_labels, predicted_labels)
    print(f"accuracy: {accuracy_score(true_labels, predicted_labels):.2f} \n")
    print(f"auc: {roc_auc_score(true_labels, predicted_scores[:,1]):.2f} \n")
    print(f"precision: {precision_score(true_labels, predicted_labels):.2f}")
    print(
        f"sensitivity/recall: = {recall_score(true_labels, predicted_labels):.2f}"
    )
    print(f"specificity: {s['specificity']:.2f}")
    print(
        "confusion matrix: \n",
        confusion_matrix(true_labels, predicted_labels),
    )


def calculate_sensitivity_and_specificity(y_true, y_predicted, positive_condition=1):
    """
    Returns sensitivity and specificity

    Parameters
    ----------
    y_true: array-like
        True class labels.
    y_predicted: array-like
        Predicted class labels. Same length as y_true
    positive_condition: int
        Default: 1.  What class label will count as "condition positive".  All other class
        labels besides this one will be counted as "condition negative".

    Notes
    -----
    All condition categories that do not match the positive condition are grouped together as one negative condition
    (relevant for multi-class classifications).
    If there are no condition-positive instances, the sensititivty will be nan
    If there are no condition-negative instances, the specificity will be nan
    """

    y_predicted = np.array(y_predicted)
    y_true = np.array(y_true)
    condition_positive_count = (y_true == positive_condition).sum()
    condition_negative_count = y_true.size - condition_positive_count
    true_positive_count = (
        (y_true == positive_condition) & (y_predicted == positive_condition)
    ).sum()
    true_negative_count = (
        (y_true != positive_condition) & (y_predicted != positive_condition)
    ).sum()
    if condition_positive_count == 0:
        sensitivity = np.nan
    else:
        sensitivity = true_positive_count / condition_positive_count
    if condition_negative_count == 0:
        specificity = np.nan
    else:
        specificity = true_negative_count / condition_negative_count
    return {"sensitivity": sensitivity, "specificity": specificity}


def train_test_split_impute_scale_resample(df, feature_column_list, target_var, train_ind, test_ind, scaler=None, imputer=None, undersample=False, use_smote=False, smote_k_neighbors=5):
    """
    Performs train/test split with given indices, imputation, and scaling, and possibly undersampling/smote

    Parameters
    ----------
    df: DataFrame
        A dataframe whose columns include the features named in feature_column_list and the outcome variable
        specified by target_var
    
    feature_column_list: array-like
        A list of strings, each a feature column of df that will go into predicting target_var
    
    target_var: str
        The name of a column of df that corresponds to the target or outcome variable (whose values could be discrete
        as in a classification task, or continuous as in a regression task)

    train_ind:
        Array of integer-valued training set indices (for use with df iloc) 

    test_ind
        Array of integer-valued test set indices (for use with df iloc) 

    scaler: optional
        An initalized sklearn scaler, like StandardScaler() or MinMaxScaler().  Scaling will not be performed
        if scaler is None. Default=None.

    imputer: optional
        An initalized sklearn imputer, like SimpleImputer(missing_values=np.nan, strategy='mean').
        Default: none.  Imputation will be performed if imputer is not None. Default=None

    undersample: bool, optional
        If true, undersample the majority class. Default=False
        Should be false unless this is being used for classification (rather than regression).
    
    use_smote: bool, optional
        If true, use SMOTE for minority class oversampling.  Default=False
        Should be false unless this is being used for classification (rather than regression).
    
    smote_k_neighbors: int, optional
        Number of neighbors to be used the SMOTE oversampling.  Default=5.
        Irrelevant unless use_smote is True  

    Returns
    -------
    X_train: DataFrame
        Training feature data, with feature values for each data point. Columns are the features in feature_column_list.
        Each data point is a row.  A subset of df.

    X_test: DataFrame
        Test feature data, with feature values for each data point. Columns are the features in feature_column_list.  Each data point is a row.
        A subset of df.

    y_train: DataFrame
        Training target variable data.  Each row corresponds to a data point, and the DataFrame gives the value of target_var. 
        A subset of df.

    y_test: DataFrame
        Test target variable data.  Each row corresponds to a data point, and the DataFrame gives the value of target_var. 
        A subset of df.

    Notes
    -----
    Called by classification_leave_out_one_ID_cv and regression_leave_out_one_ID_cv
    """

    X_train = df[feature_column_list].iloc[train_ind]
    X_test = df[feature_column_list].iloc[test_ind]
    y_train = df[target_var].iloc[train_ind]
    y_test = df[target_var].iloc[test_ind]

    if imputer is not None:
        imp = imputer.fit(X_train)
        X_train = imp.transform(X_train)
        X_test = imp.transform(X_test) 

    if scaler is not None:
        X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=feature_column_list)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_column_list)

    if undersample: # only use if this is a classification problem
        us = RandomUnderSampler(sampling_strategy=1/2)
        X_train, y_train = us.fit_resample(X_train,y_train)

    if use_smote: # only use if this is a classification problem
        sm = SMOTE(k_neighbors=smote_k_neighbors)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    X_train = pd.DataFrame(X_train, columns=feature_column_list)
    X_test = pd.DataFrame(X_test, columns=feature_column_list)

    return X_train, X_test, y_train, y_test


def classification_leave_out_one_ID_cv(df, column_list, clf_base, target_var='gen_diagnosis_num', scaler=None, imputer=None, undersample = False, use_smote=False,  column_selection='', feature_corr_max_between_features=1, feature_corr_min_with_target=0, column_select_max_features=10, record_features=False, record_misclassifications=True, verbose=True):  
    '''
    Perform classification, leaving out the session(s) from ID at a time, and return resulting predictions and classification scores.

    Parameters
    ----------
    df: DataFrame
        A DataFrame that includes the feature columns given in column_list, the column with name target_variable that 
        is the classification target variable (default='gen_diagnosis_num'), and an 'ID' column with the ID
        that is specific to one subject (and may correspond to multiple sessions, for repeat visitors).

    column_list: array (str)
        A list of strings, each a column name in df. These (or a subset of column_selection is specified),
        are the features that will be used to train the classification model and predict the classification
        
    clf_base: 
        An initialized sklearn classifier

    target_var: str, optional
        Default='gen_diagnosis_num, The name of the column of df that gives the classification target variable.

    scaler: optional
        An initalized sklearn scaler, like StandardScaler() or MinMaxScaler().  Scaling will not be performed
        if scaler is None. Default=None.

    imputer: optional
        An initalized sklearn imputer, like SimpleImputer(missing_values=np.nan, strategy='mean').
        Default: none.  Imputation will be performed if imputer is not None. Default=None

    undersample: bool, optional
        If true, undersample the majority class. Default=False
    
    use_smote: bool, optional
        If true, use SMOTE for minority class oversampling.  Default=False

    column_selection {'', 'rf', 'feature_correlations'}
        Default=''.  If a valid nonempty selection, this specifies an approach to use for pre-selection of
        features before classification.  'rf' indicates that the a preliminary random forest will be trained
        and the features with the highest importances will be retained (up to a quantity of column_select_max_features total features)
        A selection of 'feature_correlations's means that features will be iteratively removed that have too high of a correlation
        with each other (above feature_corr_max_between_features) or two low of a correlation with the target (below feature_corr_min_with_target)
        Feature selection is based on the training set for each fold.

    feature_corr_max_between_features: float, optional
        Default=1. If column_selection='feature_correlations', then a feature selection step before the model fitting
        will iteratively remove features that have a correlation with each other that is greater than this value
        in absolute value (the feature in each highly correlated pair that is retained is the one with higher correlation
        to the target variable).  Selection is made based only on the training set.  Irrelevant unless column_selection='feature_correlations'.
    
    feature_corr_min_with_target: float, optional
        Default=0. If column_selection='feature_correlations', then a feature selection step before the model fitting
        will iteratively remove features that have a correlation with the target that is less than than this value
        in absolute value. Selection is made based only on the training set.  Irrelevant unless column_selection='feature_correlations'.
    
    column_select_max_features: int, optional
        Default=10.  If column_selection='rf', this is the maximum number of features that will be retained for
        classification after the feature selection step.
    
    record_features: bool, optional
        Default=False.  If True, record the feature importances for the classifier and include them in the results
        (Applies if clf_base is a random-forest based model).
     
    verbose: bool, optional
        If True, print the true and predicted classification values for each test-set ID as the predictions are made.
        Default=True

    Returns
    -------
    results: dict
        Lists of: accuracy scores (scores_avg', 'score_med'), Training accuracy scores('training_scores_avg','training_scores_med': statistics.median(train_scores))
        Misclassification sessions ('misclassifications') and their predicted probabilities of a positive classification under the model ('misclassification_prediction_probs')
        A confusion matrix, 
        A list of all true and predicted classifications ('y_true_all', 'y_predicted_all',
        the associated model-assigned probabilities ('predict_probs_all'), and the sessions for each
        A list of the unique class labels in the data 'class_label_list' 
    
    last_train_test_split: dict
        Includes the train/test indices, features and test values, and model predicted classifications and
        probabilities for the last train/test split in the loop.

    Notes
    -----
    Calls train_test_split_impute_scale_resample to perform any over/undersampling (including SMOTE), scaling, imputing
    '''
    
    # Initialize results variables
    results = {}
    y_true_all, y_predicted_all, predict_probs_all = [],[],[]
    scores , train_scores, predict_probs_misclass = [],[],[]
    sessions, misclassifications = [], []
    labels = [x for x in set(df[target_var])]
    num_categories = len(labels)
    summed_confusion_matrix = np.zeros((num_categories, num_categories))

    if record_features:
        features_sum = pd.Series(index=column_list)

    logo = LeaveOneGroupOut()
    logo.get_n_splits(df[column_list], df[target_var],groups=df['ID'])

    for train_ind, test_ind in logo.split(df[column_list], df[target_var],groups=df['ID']):      
        clf_selected = clone(clf_base)
        X_train, X_test, y_train, y_test = train_test_split_impute_scale_resample(df, column_list, target_var, train_ind, test_ind, scaler, imputer, undersample, use_smote)

        selected_columns = column_list
        if column_selection == 'rf':
            clf = SelectFromModel(BalancedRandomForestClassifier(n_estimators=200, min_samples_leaf=6, max_depth=6), max_features=min([column_select_max_features,len(column_list)]))
            clf.fit(X_train, y_train)
            selected_columns = X_train.columns[clf.get_support()]
        elif column_selection == 'feature_correlations':
            X_train, selected_columns = greedily_remove_features_by_correlation_with_target(X_train, y_train, min_correlation=feature_corr_min_with_target, verbose=False)
            X_train, selected_columns =  greedily_remove_features_by_correlation_with_each_other(X_train, y_train, max_correlation=feature_corr_max_between_features, verbose=False)                
                    
        clf_selected.fit(X_train[selected_columns], y_train)
        y_predicted = clf_selected.predict(X_test[selected_columns])
        predict_probs = clf_selected.predict_proba(X_test[selected_columns])

        y_true_all += list(y_test)
        y_predicted_all += list(y_predicted)
        predict_probs_all += list(predict_probs)
        sessions += list(df.index.values[test_ind])
 
        mistakes = y_predicted != y_test
        mistake_inds = test_ind[mistakes]
        misclassifications = misclassifications + list(df['ID'].index.values[mistake_inds]) 
        predict_probs_misclass.extend(list( predict_probs[mistakes,:]))

        scores.append(clf_selected.score(X_test[selected_columns],y_test))
        train_scores.append(clf_selected.score(X_train[selected_columns],y_train))
        
        if record_features:
            features_curr = pd.Series(clf_selected.feature_importances_, index=selected_columns)
            features_sum = features_sum.add(features_curr, fill_value=0)

        summed_confusion_matrix += confusion_matrix(y_test, y_predicted,labels=labels)   

        if verbose:
            print(list(y_test),y_predicted)

    results = {
        'scores_avg':sum(scores)/len(scores),
        'score_med':statistics.median(scores),
        'training_scores_avg': sum(train_scores)/len(train_scores),
        'training_scores_med': statistics.median(train_scores),
        'misclassifications': misclassifications,
        'summed_confusion_matrix': summed_confusion_matrix,
        'class_label_list': labels,
        'y_predicted_all' : y_predicted_all,
        'predict_probs_all' : predict_probs_all,
        'y_true_all' : y_true_all,
        'sessions' : sessions,
        'misclassification_prediction_probs' : predict_probs_misclass
    }
 
    if record_features:
        results['feature_imps'] = features_sum.sort_values(ascending=False) / len(scores)

    last_train_test_split = {}
    last_train_test_split['X_train'] = X_train[selected_columns]
    last_train_test_split['X_test'] = X_test[selected_columns]
    last_train_test_split['y_test'] = y_test
    last_train_test_split['y_train'] = y_train
    last_train_test_split['predict_probs'] = predict_probs
    last_train_test_split['clf'] = clf_selected
    last_train_test_split['selected_columns'] = selected_columns
    return results, last_train_test_split


def regression_leave_out_one_ID_cv(df, column_list, target_var, reg_base, scaler=None, imputer=None, column_selection=False, feature_corr_max_between_features=1, feature_corr_min_with_target=0, column_select_max_features=10, record_features=False, use_smote=False, verbose=False):
    '''
    Perform regression, leaving out the session(s) from ID at a time, and return resulting predictions

    Parameters
    ----------
    df: DataFrame
        A DataFrame that includes the feature columns given in column_list, the column with name target_variable that 
        is the regression target variable (default='gen_diagnosis_num'), and an 'ID' column with the ID
        that is specific to one subject (and may correspond to multiple sessions, for repeat visitors).

    column_list: array (str)
        A list of strings, each a column name in df. These (or a subset of column_selection is specified),
        are the features that will be used to train the regression model and predict the target variable
        
    target_var: str
        The name of the column of df that gives the regression target variable.

    reg_base:
        An initialized sklearn regression object

    scaler: optional
        An initalized sklearn scaler, like StandardScaler() or MinMaxScaler().  Scaling will not be performed
        if scaler is None. Default=None.

    imputer: optional
        An initalized sklearn imputer, like SimpleImputer(missing_values=np.nan, strategy='mean').
        Default: none.  Imputation will be performed if imputer is not None. Default=None

    column_selection {'', 'rf', 'feature_correlations'}
        Default=''.  If a valid nonempty selection, this specifies an approach to use for pre-selection of
        features before regression.  'rf' indicates that the a preliminary random forest will be trained
        and the features with the highest importances will be retained (up to a quantity of column_select_max_features total features)
        A selection of 'feature_correlations's means that features will be iteratively removed that have too high of a correlation
        with each other (above feature_corr_max_between_features) or two low of a correlation with the target (below feature_corr_min_with_target)
        Feature selection is based on the training set for each fold.

    feature_corr_max_between_features: float, optional
        Default=1. If column_selection='feature_correlations', then a feature selection step before the model fitting
        will iteratively remove features that have a correlation with each other that is greater than this value
        in absolute value (the feature in each highly correlated pair that is retained is the one with higher correlation
        to the target variable).  Selection is made based only on the training set.  Irrelevant unless column_selection='feature_correlations'.
    
    feature_corr_min_with_target: float, optional
        Default=0. If column_selection='feature_correlations', then a feature selection step before the model fitting
        will iteratively remove features that have a correlation with the target that is less than than this value
        in absolute value. Selection is made based only on the training set.  Irrelevant unless column_selection='feature_correlations'.
    
    column_select_max_features: int, optional
        Default=10.  If column_selection='rf', this is the maximum number of features that will be retained for
        regression after the feature selection step.
    
    record_features: bool, optional
        Default=False.  If True, record the feature importances for the regressor and include them in the results
        (Applies if reg_base is a random-forest based model).
     
    verbose: bool, optional
        If True, print the true and predicted regression values for each test-set ID as the predictions are made.
        Default=True

    use_smote: bool, optional 
        Default=False.  See commented-out code for including this step; SMOTE is potentially applicable if trying to 
        perform regression for something like arm_BARS, where there are only a few possible outcome variabes, so one
        could treat these as classifications and resample (oversample) before regressing.

    Returns
    -------
    results: dict
        'mae' (mean_absolute_error), 'r_squared', 'y_true_all' 'y_predicted_all', 
        'median_num_features': median number of features left after the optional feature preselection step
        'diagnoses': list of diagnosis nums, indices corresponding to y_true_all and y_predicted all.
        'sessions': list of session strings, indices corresponding to y_true_all and y_predicted all.
        
    
    last_train_test_split: dict
        Includes the train/test indices, features and test values, and the trained regression model
        for the last train/test split in the loop.

    Notes
    -----
    Calls train_test_split_impute_scale_resample to perform any imputing, scaling
    '''
    
    y_true_all, y_predicted_all, num_features, diagnosis_of_y_true, sessions = [], [], [], [], []
    
    if record_features:
        features_sum = pd.Series(index=column_list)
    
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(df[column_list], df[target_var], groups=df['ID'])
    for train_ind, test_ind in logo.split(df[column_list], df[target_var], groups=df['ID']):

        reg_selected = clone(reg_base)

        # if use_smote: # make this into integer classes 
        #     df[target_var] = [int(score*2) for score in list(df[target_var])]


        X_train, X_test, y_train, y_test = train_test_split_impute_scale_resample(df, column_list, target_var, train_ind, test_ind, scaler, imputer, use_smote=use_smote, smote_k_neighbors=2)

        # if use_smote:
        #     df[target_var] = df[target_var] / 2


        selected_columns = column_list
        if column_selection == 'rf':
            reg = SelectFromModel(RandomForestRegressor(n_estimators=200, min_samples_leaf=6, max_depth=6), max_features=min([column_select_max_features,len(column_list)]))
            reg.fit(X_train, y_train)
            selected_columns = X_train.columns[reg.get_support()]
        elif column_selection == 'feature_correlations':
            X_train, selected_columns = greedily_remove_features_by_correlation_with_target(X_train, y_train, min_correlation=feature_corr_min_with_target, verbose=False)
            X_train, selected_columns =  greedily_remove_features_by_correlation_with_each_other(X_train, y_train, max_correlation=feature_corr_max_between_features, verbose=False)                
                    
        reg_selected.fit(X_train[selected_columns], y_train)
        y_predicted = reg_selected.predict(X_test[selected_columns])

        # Append true and predicted values
        num_features.append(len(selected_columns))
        y_true_all += list(y_test)
        y_predicted_all += list(y_predicted)
        diagnosis_of_y_true += list(df['gen_diagnosis_num'].iloc[test_ind])
        sessions += list(df.index.values[test_ind])
        
        if record_features:
            features_curr = pd.Series(reg_selected.feature_importances_, index=selected_columns)
            features_sum = features_sum.add(features_curr, fill_value=0)

        if verbose:
            print(y_test, y_predicted)
    
    results = {
        'mae': mean_absolute_error(y_true_all, y_predicted_all),
        'r_squared': r2_score(y_true_all, y_predicted_all),
        'y_true_all': y_true_all,
        'y_predicted_all': y_predicted_all,
        'median_num_features': statistics.median(num_features),
        'diagnoses': diagnosis_of_y_true,
        'sessions': sessions
    }

    if record_features:
        results['feature_imps'] = features_sum.sort_values(ascending=False) / n_splits

    last_train_test_split = {
        'X_train':X_train[selected_columns],
        'X_test': X_test[selected_columns],
        'y_test': y_test,
        'y_train':y_train,
        'reg': reg_selected,
        'selected_columns': selected_columns
    }

    return results, last_train_test_split


def classification_leave_out_one_ID_cv_with_hyperparams(df, column_list, clf_base, target_var='gen_diagnosis_num', scaler=None, imputer=None, undersample = False, use_smote=False,  column_selection='', feature_corr_max_between_features=1, feature_corr_min_with_target=0, column_select_max_features=10, record_features=False, record_misclassifications=True, verbose=True):  
    '''
    Perform classification, leaving out the session(s) from ID at a time, and return resulting predictions and classification scores.

    Parameters
    ----------
    df: DataFrame
        A DataFrame that includes the feature columns given in column_list, the column with name target_variable that 
        is the classification target variable (default='gen_diagnosis_num'), and an 'ID' column with the ID
        that is specific to one subject (and may correspond to multiple sessions, for repeat visitors).

    column_list: array (str)
        A list of strings, each a column name in df. These (or a subset of column_selection is specified),
        are the features that will be used to train the classification model and predict the classification
        
    clf_base: 
        An initialized sklearn classifier

    target_var: str, optional
        Default='gen_diagnosis_num, The name of the column of df that gives the classification target variable.

    scaler: optional
        An initalized sklearn scaler, like StandardScaler() or MinMaxScaler().  Scaling will not be performed
        if scaler is None. Default=None.

    imputer: optional
        An initalized sklearn imputer, like SimpleImputer(missing_values=np.nan, strategy='mean').
        Default: none.  Imputation will be performed if imputer is not None. Default=None

    undersample: bool, optional
        If true, undersample the majority class. Default=False
    
    use_smote: bool, optional
        If true, use SMOTE for minority class oversampling.  Default=False

    column_selection {'', 'rf', 'feature_correlations'}
        Default=''.  If a valid nonempty selection, this specifies an approach to use for pre-selection of
        features before classification.  'rf' indicates that the a preliminary random forest will be trained
        and the features with the highest importances will be retained (up to a quantity of column_select_max_features total features)
        A selection of 'feature_correlations's means that features will be iteratively removed that have too high of a correlation
        with each other (above feature_corr_max_between_features) or two low of a correlation with the target (below feature_corr_min_with_target)
        Feature selection is based on the training set for each fold.

    feature_corr_max_between_features: float, optional
        Default=1. If column_selection='feature_correlations', then a feature selection step before the model fitting
        will iteratively remove features that have a correlation with each other that is greater than this value
        in absolute value (the feature in each highly correlated pair that is retained is the one with higher correlation
        to the target variable).  Selection is made based only on the training set.  Irrelevant unless column_selection='feature_correlations'.
    
    feature_corr_min_with_target: float, optional
        Default=0. If column_selection='feature_correlations', then a feature selection step before the model fitting
        will iteratively remove features that have a correlation with the target that is less than than this value
        in absolute value. Selection is made based only on the training set.  Irrelevant unless column_selection='feature_correlations'.
    
    column_select_max_features: int, optional
        Default=10.  If column_selection='rf', this is the maximum number of features that will be retained for
        classification after the feature selection step.
    
    record_features: bool, optional
        Default=False.  If True, record the feature importances for the classifier and include them in the results
        (Applies if clf_base is a random-forest based model).
     
    verbose: bool, optional
        If True, print the true and predicted classification values for each test-set ID as the predictions are made.
        Default=True

    Returns
    -------
    results: dict
        Lists of: accuracy scores (scores_avg', 'score_med'), Training accuracy scores('training_scores_avg','training_scores_med': statistics.median(train_scores))
        Misclassification sessions ('misclassifications') and their predicted probabilities of a positive classification under the model ('misclassification_prediction_probs')
        A confusion matrix, 
        A list of all true and predicted classifications ('y_true_all', 'y_predicted_all',
        the associated model-assigned probabilities ('predict_probs_all'), and the sessions for each
        A list of the unique class labels in the data 'class_label_list' 
    
    last_train_test_split: dict
        Includes the train/test indices, features and test values, and model predicted classifications and
        probabilities for the last train/test split in the loop.

    Notes
    -----
    Calls train_test_split_impute_scale_resample to perform any over/undersampling (including SMOTE), scaling, imputing
    '''
    
    # Initialize results variables
    results = {}
    y_true_all, y_predicted_all, predict_probs_all = [],[],[]
    scores , train_scores, predict_probs_misclass = [],[],[]
    sessions, misclassifications = [], []
    labels = [x for x in set(df[target_var])]
    num_categories = len(labels)
    summed_confusion_matrix = np.zeros((num_categories, num_categories))

    if record_features:
        features_sum = pd.Series(index=column_list)

    logo = LeaveOneGroupOut()
    logo.get_n_splits(df[column_list], df[target_var],groups=df['ID'])

    min_samples_leaf = [1, 2, 4]
    max_depth = [4, 6, 8]
    random_grid = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}

    for train_ind, test_ind in logo.split(df[column_list], df[target_var],groups=df['ID']):      
        #clf_selected = clone(clf_base)


        X_train, X_test, y_train, y_test = train_test_split_impute_scale_resample(df, column_list, target_var, train_ind, test_ind, scaler, imputer, undersample, use_smote)

        rf = BalancedRandomForestClassifier()

        grid_search = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 3, n_jobs=-1)
        # Fit the random search model
        grid_search.fit(X_train, y_train)
        params = grid_search.best_params_
        clf_selected = BalancedRandomForestClassifier(n_estimators=200, **params)

        selected_columns = column_list
        if column_selection == 'rf':
            clf = SelectFromModel(BalancedRandomForestClassifier(n_estimators=200, min_samples_leaf=6, max_depth=6), max_features=min([column_select_max_features,len(column_list)]))
            clf.fit(X_train, y_train)
            selected_columns = X_train.columns[clf.get_support()]
        elif column_selection == 'feature_correlations':
            X_train, selected_columns = greedily_remove_features_by_correlation_with_target(X_train, y_train, min_correlation=feature_corr_min_with_target, verbose=False)
            X_train, selected_columns =  greedily_remove_features_by_correlation_with_each_other(X_train, y_train, max_correlation=feature_corr_max_between_features, verbose=False)                
                    
        clf_selected.fit(X_train[selected_columns], y_train)
        y_predicted = clf_selected.predict(X_test[selected_columns])
        predict_probs = clf_selected.predict_proba(X_test[selected_columns])

        y_true_all += list(y_test)
        y_predicted_all += list(y_predicted)
        predict_probs_all += list(predict_probs)
        sessions += list(df.index.values[test_ind])
 
        mistakes = y_predicted != y_test
        mistake_inds = test_ind[mistakes]
        misclassifications = misclassifications + list(df['ID'].index.values[mistake_inds]) 
        predict_probs_misclass.extend(list( predict_probs[mistakes,:]))

        scores.append(clf_selected.score(X_test[selected_columns],y_test))
        train_scores.append(clf_selected.score(X_train[selected_columns],y_train))
        
        if record_features:
            features_curr = pd.Series(clf_selected.feature_importances_, index=selected_columns)
            features_sum = features_sum.add(features_curr, fill_value=0)

        summed_confusion_matrix += confusion_matrix(y_test, y_predicted,labels=labels)   

        if verbose:
            print(list(y_test),y_predicted)

    results = {
        'scores_avg':sum(scores)/len(scores),
        'score_med':statistics.median(scores),
        'training_scores_avg': sum(train_scores)/len(train_scores),
        'training_scores_med': statistics.median(train_scores),
        'misclassifications': misclassifications,
        'summed_confusion_matrix': summed_confusion_matrix,
        'class_label_list': labels,
        'y_predicted_all' : y_predicted_all,
        'predict_probs_all' : predict_probs_all,
        'y_true_all' : y_true_all,
        'sessions' : sessions,
        'misclassification_prediction_probs' : predict_probs_misclass
    }
 
    if record_features:
        results['feature_imps'] = features_sum.sort_values(ascending=False) / len(scores)

    last_train_test_split = {}
    last_train_test_split['X_train'] = X_train[selected_columns]
    last_train_test_split['X_test'] = X_test[selected_columns]
    last_train_test_split['y_test'] = y_test
    last_train_test_split['y_train'] = y_train
    last_train_test_split['predict_probs'] = predict_probs
    last_train_test_split['clf'] = clf_selected
    last_train_test_split['selected_columns'] = selected_columns
    return results, last_train_test_split
