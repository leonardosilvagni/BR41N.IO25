#%%
from preprocessing import *

# Some standard pythonic imports
import warnings
warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt

# MNE functions
from mne import Epochs, find_events
from mne.decoding import Vectorizer

# Scikit-learn and Pyriemann ML functionalities
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from pyriemann.estimation import ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from scipy.stats import skew, kurtosis

# %%


# Classifier dictionary
clfs = OrderedDict()
clfs['Vect + LR'] = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression())
clfs['Vect + RegLDA'] = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
clfs['XdawnCov + TS'] = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(), LogisticRegression())
clfs['XdawnCov + MDM'] = make_pipeline(XdawnCovariances(estimator='oas'), MDM())
clfs['ERPCov + TS'] = make_pipeline(ERPCovariances(), TangentSpace(), LogisticRegression())
clfs['ERPCov + MDM'] = make_pipeline(ERPCovariances(), MDM())
clfs['Vect + SVM'] = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel='rbf', C=3, probability=True))
clfs['Vect + KNN'] = make_pipeline(Vectorizer(), StandardScaler(), KNeighborsClassifier())
clfs['Vect + Bagging Tree'] = make_pipeline(Vectorizer(), BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, max_samples=0.8, random_state=42))
clfs['Stat + Fisher'] = make_pipeline(StandardScaler(), FisherDiscriminantClassifier())

# Loop over subjects
subject_ids = [1, 2, 3, 4, 5]
folder = '/Users/magbi/BR41N.IO25/p300-speller'
all_results = []

for sid in subject_ids:
    print(f"\nLoading subject S{sid}")
    filepath = os.path.join(folder, f"S{sid}.mat")
    epochs = get_epochs_from_file(filepath)
    epochs = epochs.crop(tmin= -0.1, tmax=0.8)
    epochs.pick_types(eeg=True)
    X = epochs.get_data() * 1e6
    y = (epochs.events[:, -1] == 1).astype(int)

    # Balance dataset
    pos_idx = np.where(y == 1)[0][:1000]
    neg_idx = np.where(y == 0)[0][:150]
    idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    X = X[idx]
    y = y[idx]

    X_stat = extract_stat_features(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in clfs.items():
        print(f"Training {name} on S{sid}")
        X_input = X_stat if "Stat" in name else X

        try:
            y_score = cross_val_predict(clf, X_input, y, cv=cv, method='decision_function')
        except:
            try:
                y_score = cross_val_predict(clf, X_input, y, cv=cv, method='predict_proba')[:, 1]
            except:
                y_score = cross_val_predict(clf, X_input, y, cv=cv)

        y_pred = (y_score > 0.5).astype(int)
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_score)
        cm = confusion_matrix(y, y_pred)

        all_results.append({"Subject": sid, "Method": name, "Accuracy": acc, "AUC": auc, "ConfusionMatrix": cm})
        print(f"Subject {sid}, {name} - Acc: {acc:.3f}, AUC: {auc:.3f}")
        

# Convert to DataFrame
results_df = pd.DataFrame(all_results)
grouped = results_df.groupby("Method").agg({"AUC": ["mean", "std"], "Accuracy": ["mean", "std"]})
grouped.columns = ['AUC_mean', 'AUC_std', 'Acc_mean', 'Acc_std']
grouped = grouped.reset_index()

# Plotting
plt.figure(figsize=(10, 5))
plt.barh(grouped['Method'], grouped['AUC_mean'], xerr=grouped['AUC_std'], color='skyblue', edgecolor='black')
plt.xlabel("Mean AUC")
plt.title("AUC Scores Across Subjects")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(grouped['Method'], grouped['Acc_mean'], xerr=grouped['Acc_std'], color='lightgreen', edgecolor='black')
plt.xlabel("Mean Accuracy")
plt.title("Accuracy Scores Across Subjects")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
print("\nBest accuracy per subject:")
for sid in subject_ids:
    subject_data = results_df[results_df['Subject'] == sid]
    best_row = subject_data.loc[subject_data['Accuracy'].idxmax()]
    print(f"Subject {sid}: {best_row['Method']} with accuracy {best_row['Accuracy']:.3f}")

# %%

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize an empty list to hold metric rows
metrics = []

# Loop through each row of your results DataFrame
for row in results_df.itertuples():
    cm = row.ConfusionMatrix
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        accuracy = precision = recall = f1 = 0

    metrics.append({
        "Subject": row.Subject,
        "Method": row.Method,
        "Accuracy": accuracy,
        "AUC": row.AUC,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics)

# Assuming metrics_df is already created as shown earlier
best_per_subject = metrics_df.loc[metrics_df.groupby("Subject")["Accuracy"].idxmax()].reset_index(drop=True)

# Display the table
print(best_per_subject)

# %%
