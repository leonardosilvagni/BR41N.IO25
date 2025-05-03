#%%
from feature_extraction import *
from preprocessing import *
from models.tvlda import TVLDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import numpy as np
import torch
from visualization import plot_metrics
from test_model import evaluate_model
filename = os.path.join('p300-speller','S1.mat')

epochs = get_epochs_from_file(filename)
features_TVLDA = extract_features_TVLDA(epochs)
#%%
from mne.decoding import Vectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# Scikit-learn and Pyriemann ML functionalities
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
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
clfs = OrderedDict()
clfs['Vect + LR'] = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression())
clfs['Vect + RegLDA'] = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
#clfs['Xdawn + RegLDA'] = make_pipeline(XdawnWrapper(2, classes=[1]), Vectorizer(), LDA(shrinkage='auto', solver='eigen'))

clfs['XdawnCov + TS'] = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(), LogisticRegression())
clfs['XdawnCov + MDM'] = make_pipeline(XdawnCovariances(estimator='oas'), MDM())


clfs['ERPCov + TS'] = make_pipeline(ERPCovariances(), TangentSpace(), LogisticRegression())
clfs['ERPCov + MDM'] = make_pipeline(ERPCovariances(), MDM())
# format data
clfs['TVLDA'] = TVLDA()
clfs['LDA'] = LinearDiscriminantAnalysis()

#%%
for m in clfs:
    print(f'Running {m}')
    conf_matr, auc = evaluate_model(features_TVLDA, clfs[m], n_splits=5, windows=True)
    plot_metrics(conf_matr, auc)

# %%
