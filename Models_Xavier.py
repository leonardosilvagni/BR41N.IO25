#%%
from preprocessing import *

# Some standard pythonic imports
import warnings
warnings.filterwarnings('ignore')
import os,numpy as np,pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt

# MNE functions
from mne import Epochs,find_events
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
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import numpy as np

# Fisher custom estimator
class FisherDiscriminantClassifier(BaseEstimator):
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

    def predict_proba(self, X):
        z = X @ self.w
        proba = (z - z.min()) / (z.max() - z.min() + 1e-9)
        return np.vstack([1 - proba, proba]).T

#%%
def extract_stat_features(X):
    feats = []
    for trial in X:
        trial_feat = []
        for ch in trial:
            AAM = np.max(np.abs(ch))
            mu = np.mean(ch)
            std = np.std(ch, ddof=1)
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

#%% define classifiers
clfs = OrderedDict()
clfs['Vect + LR'] = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression())
clfs['Vect + RegLDA'] = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
clfs['XdawnCov + TS'] = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(), LogisticRegression())
clfs['XdawnCov + MDM'] = make_pipeline(XdawnCovariances(estimator='oas'), MDM())
clfs['ERPCov + TS'] = make_pipeline(ERPCovariances(), TangentSpace(), LogisticRegression())
clfs['ERPCov + MDM'] = make_pipeline(ERPCovariances(), MDM())
clfs['Vect + SVM'] = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel='rbf', C=3, gamma='scale'))
clfs['Vect + KNN'] = make_pipeline(Vectorizer(), StandardScaler(), KNeighborsClassifier())
clfs['Vect + Bagging Tree'] = make_pipeline(Vectorizer(), BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators=50, 
    max_samples=0.8, 
    random_state=42
))
clfs['Stat + Fisher'] = make_pipeline(StandardScaler(), FisherDiscriminantClassifier())

#%% loop for all subjects
subject_ids = [1, 2, 3, 4, 5]
folder = '/Users/magbi/BR41N.IO25/p300-speller'

all_results = []

for sid in subject_ids:
    filepath = os.path.join(folder, f"S{sid}.mat")
    print(f"\nLoading subject {sid} from {filepath}")
    epochs = get_epochs_from_file(filepath)
    epochs.pick_types(eeg=True)
    X = epochs.get_data() * 1e6
    y = (epochs.events[:, -1] == 1).astype(int)

    # Balance dataset
    event_idx = np.where(y == 1)[0]
    non_event_idx = np.where(y == 0)[0]
    np.random.shuffle(event_idx)
    np.random.shuffle(non_event_idx)
    event_idx = event_idx[:150]
    non_event_idx = non_event_idx[:150]
    idx = np.sort(np.concatenate([event_idx, non_event_idx]))
    X = X[idx]
    y = y[idx]

    X_stat = extract_stat_features(X)
    X_flat = X.reshape(X.shape[0], -1)
    cv = StratifiedKFold(n_splits=11, shuffle=True, random_state=42)

    for m in clfs:
        print(f'Running {m} for Subject {sid}')
        X_input = X_stat if 'Stat' in m else X
        res_auc = cross_val_score(clfs[m], X_input, y, scoring='roc_auc', cv=cv, n_jobs=-1)
        res_acc = cross_val_score(clfs[m], X_input, y, scoring='accuracy', cv=cv, n_jobs=-1)
        for auc_val, acc_val in zip(res_auc, res_acc):
            all_results.append({'Subject': sid, 'Method': m, 'AUC': auc_val, 'Accuracy': acc_val})

#%% Aggregate results
results_df = pd.DataFrame(all_results)
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='AUC', y='Method', errorbar='sd')
plt.title("AUC Scores Across Subjects")
plt.xlim(0.5, 1.0)
plt.grid(True)
sns.despine()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='Accuracy', y='Method', errorbar='sd')
plt.title("Accuracy Scores Across Subjects")
plt.xlim(0.5, 1.0)
plt.grid(True)
sns.despine()
plt.tight_layout()
plt.show()

# %%
