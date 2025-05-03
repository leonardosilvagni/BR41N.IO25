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
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM


#filename = os.path.join('/Users/magbi/BR41N.IO25/p300-speller','S1.mat')

subject_ids = [1, 2, 3, 4, 5]  # or however many subjects you have
folder = '/Users/magbi/BR41N.IO25/p300-speller'


all_X = []
all_y = []
all_time = []
for sid in subject_ids:
    filepath = os.path.join(folder, f"S{sid}.mat")
    print(f"Loading subject {sid} from {filepath}")
    epochs = get_epochs_from_file(filepath)
    epochs.pick_types(eeg=True)
    X = epochs.get_data() * 1e6
    times = epochs.times
    y = epochs.events[:, -1]
    all_X.append(X)
    all_y.append(y)
    all_time.append(times)

X = np.concatenate(all_X, axis=0)  # shape: (n_epochs_total, n_channels, n_times)
y = np.concatenate(all_y, axis=0)  # shape: (n_epochs_total,)
    
#epochs = get_epochs_from_file(filename)
#%%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import numpy as np

class XdawnWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, classes=[1]):
        self.n_components = n_components
        self.classes = classes
        self.xdawn = Xdawn(self.n_components, classes=self.classes)
        
    def fit(self, X, y=None):
        self.xdawn.fit(X, y)
        return self
        
    def transform(self, X):
        return self.xdawn.transform(X)
#%%
clfs = OrderedDict()
clfs['Vect + LR'] = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression())
clfs['Vect + RegLDA'] = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
#clfs['Xdawn + RegLDA'] = make_pipeline(XdawnWrapper(2, classes=[1]), Vectorizer(), LDA(shrinkage='auto', solver='eigen'))

clfs['XdawnCov + TS'] = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(), LogisticRegression())
clfs['XdawnCov + MDM'] = make_pipeline(XdawnCovariances(estimator='oas'), MDM())


clfs['ERPCov + TS'] = make_pipeline(ERPCovariances(), TangentSpace(), LogisticRegression())
clfs['ERPCov + MDM'] = make_pipeline(ERPCovariances(), MDM())
# format data

#%%
# Balance dataset: Select 150 event samples and 150 non-event samples


#%%
# Identify indices for each class (assumes events==1 are event samples)
event_idx = np.where(y == 1)[0]
non_event_idx = np.where(y != 1)[0]

# Randomly sample 150 indices from each class
# (Assumes there are at least 150 samples in each class)
np.random.shuffle(event_idx)
np.random.shuffle(non_event_idx)
event_idx = event_idx[:150]
non_event_idx = non_event_idx[:150]

# Combine and sort indices
idx = np.sort(np.concatenate([event_idx, non_event_idx]))

# Subset X and y accordingly
X = X[idx]
y = y[idx]


# define cross validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# run cross validation for each pipeline
auc = []
accuracy = []
methods = []
conf_matrices = {}

for m in clfs:
    print(f'Running {m}')
    # Compute ROC AUC scores
    res_auc = cross_val_score(clfs[m], X, y==1, scoring='roc_auc', cv=cv, n_jobs=-1)
    auc.extend(res_auc)
    methods.extend([m] * len(res_auc))
    
    # Compute accuracy scores
    res_acc = cross_val_score(clfs[m], X, y==1, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy.extend(res_acc)
    
    # Compute confusion matrix using cross-validated predictions
    y_pred = cross_val_predict(clfs[m], X, y==1, cv=cv, n_jobs=-1)
    cm = confusion_matrix(y==1, y_pred)
    conf_matrices[m] = cm

results = pd.DataFrame(data=auc, columns=['AUC'])
results['Method'] = methods

#%%
# AUC Visualization
plt.figure(figsize=(8, 4))
sns.barplot(data=results, x='AUC', y='Method')
plt.xlim(0.92, 1)
plt.grid(True)
sns.despine()
plt.title("AUC Scores")
plt.show()

# Accuracy Visualization
results_acc = pd.DataFrame({'Accuracy': accuracy, 'Method': methods})
plt.figure(figsize=(8, 4))
sns.barplot(data=results_acc, x='Accuracy', y='Method')
plt.xlim(0.5, 1)
plt.grid(True)
sns.despine()
plt.title("Accuracy Scores")
plt.show()

# Confusion Matrices Visualization
# Determine grid size
num_methods = len(conf_matrices)
cols = 3
rows = (num_methods // cols) + (num_methods % cols > 0)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = axes.flatten()

for ax, method in zip(axes, conf_matrices):
    cm = conf_matrices[method]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(method)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

# Hide any unused subplots
for ax in axes[num_methods:]:
    ax.axis('off')

plt.tight_layout()
plt.show()
# %%
print('AUC results',auc)

# %%
for method, cm in conf_matrices.items():
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP+TN) / cm.sum()
    precision = TP / (TP+FP) if (TP+FP)>0 else 0
    recall = TP / (TP+FN) if (TP+FN)>0 else 0
    f1 = 2 * precision * recall / (precision+recall) if (precision+recall)>0 else 0
    print(f"{method}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1 Score={f1:.3f}")
# %%
