#%%
from feature_extraction import *
from preprocessing import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, confusion_matrix
import os
import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from visualization import plot_metrics
def evaluate_model(features, model, n_splits=5, windows=False):
    """
    Evaluate a model using cross validation and return the aggregated confusion matrix and AUC. Model just needs fit and predict.

    Args:
        features (dict): Dictionary with keys 'event_raw' and 'nonevent_raw' representing positive and negative events.
        model: A scikit-learn classifier with fit/predict methods.
        n_splits (int): Number of folds for cross validation.

    Returns:
        tuple: (conf_matrix, auc) where conf_matrix is a 2x2 numpy array and auc is the area under ROC curve.
    """
    print(f"Evaluating model: {model.__class__.__name__} with {n_splits} folds")
    # Convert features to torch tensors and ensure both classes have the same number of samples
    if windows:
        positive_events = torch.tensor(features['event'])
        negative_events = torch.tensor(features['nonevent'])
    else:
        positive_events = torch.tensor(features['event_raw']).reshape(features['event_raw'].shape[0], -1)
        negative_events = torch.tensor(features['nonevent_raw']).reshape(features['nonevent_raw'].shape[0], -1)
    negative_events = negative_events[:positive_events.shape[0], ...]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # For AUC computation, we aggregate true labels and decision values
    y_true_total = []
    decision_scores_total = []
    # For confusion matrix, aggregate predictions and true labels (converted to 0/1)
    y_pred_total = []
    y_class_total = []

    # Create full dataset and corresponding labels for stratification
    X = torch.cat((positive_events, negative_events), dim=0).numpy()
    y = np.concatenate((np.ones(len(positive_events)), -np.ones(len(negative_events))), axis=0)

    # Loop over cross validation folds using both X and y
    for train_idx, test_idx in kf.split(X, y):
        X_train = torch.tensor(X[train_idx])
        y_train = torch.tensor(y[train_idx])
        X_test = torch.tensor(X[test_idx])
        y_test = torch.tensor(y[test_idx])
        # Train the classifier
        model.fit(X_train, y_train)
        
        # Get predictions for confusion matrix
        predictions = model.predict(X_test)
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
            
        # Convert predictions and y_test to 0/1 labels (1 -> 1, -1 -> 0)
        y_pred_fold = (predictions == 1).astype(int)
        y_class_fold = (y_test == 1).numpy().astype(int)
        y_pred_total.extend(y_pred_fold.tolist())
        y_class_total.extend(y_class_fold.tolist())

        # Get decision scores for AUC; use decision_function if available
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            scores = model.predict_proba(X_test)[:, 1]
        y_true_total.extend(y_class_fold.tolist())
        decision_scores_total.extend(scores.tolist())
    
    # Aggregate confusion matrix
    conf_matrix = confusion_matrix(y_class_total, y_pred_total)
    # Compute AUC
    auc = roc_auc_score(y_true_total, decision_scores_total)

    return conf_matrix, auc

if __name__ == '__main__':
    filename = os.path.join('p300-speller', 'S1.mat')
    epochs = get_epochs_from_file(filename)
    features_LDA = extract_features_TVLDA(epochs)  # assuming the same features are applicable

    # Create an instance of the classifier
    lda_model = LinearDiscriminantAnalysis()
    # Evaluate the model using cross validation
    conf_matrix, auc = evaluate_model(features_LDA, lda_model, n_splits=5)

    plot_metrics(conf_matrix, auc)
# %%