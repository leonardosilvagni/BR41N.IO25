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

# Fisher custom estimator
class FisherDiscriminantClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X0 = X[y == 0]
        X1 = X[y == 1]
        self.mu0 = X0.mean(axis=0)
        self.mu1 = X1.mean(axis=0)
        S0 = np.cov(X0, rowvar=False)
        S1 = np.cov(X1, rowvar=False)
        Sw = S0 + S1
        self.w = np.linalg.pinv(Sw) @ (self.mu1 - self.mu0)
        proj0 = X0 @ self.w
        proj1 = X1 @ self.w
        self.threshold = (proj0.mean() + proj1.mean()) / 2
        return self

    def predict(self, X):
        z = X @ self.w
        return (z > self.threshold).astype(int)

    def decision_function(self, X):
        return X @ self.w

# Statistical feature extraction

def extract_stat_features(X):
    feats = []
    for trial in X:
        trial_feat = []
        for ch in trial:
            AAM = np.max(np.abs(ch))
            mu = np.mean(ch)
            std = np.std(ch)
            med = np.median(ch)
            sk = skew(ch)
            ku = kurtosis(ch)
            WL = np.sum(np.abs(np.diff(ch)))
            diff1 = np.diff(ch)
            SSC = np.sum((diff1[:-1] * diff1[1:] < 0) & (np.abs(diff1[:-1] - diff1[1:]) > 1e-6))
            P = np.mean(ch ** 2)
            E = np.sum(ch ** 2)
            trial_feat.extend([AAM, mu, std, med, sk, ku, WL, SSC, P, E])
        feats.append(trial_feat)
    return np.array(feats)

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
    epochs.pick_types(eeg=True)
    X = epochs.get_data() * 1e6
    y = (epochs.events[:, -1] == 1).astype(int)

    # Balance dataset
    pos_idx = np.where(y == 1)[0][:150]
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
