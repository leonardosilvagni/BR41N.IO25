#%%
from preprocessing import *
import warnings
warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt
import pywt
from data_augmentation import balance_data
from scipy.stats import skew, kurtosis

def extract_stat_features(signal):
    # Compute stats on a 1D array and return 8 features:
    AAM = np.max(np.abs(signal))
    mu = np.mean(signal)
    std = np.std(signal, ddof=1)
    med = np.median(signal)
    sk = skew(signal)
    ku = kurtosis(signal)
    P = np.mean(signal ** 2)
    E = np.sum(signal ** 2)
    return np.array([AAM, mu, std, med, sk, ku, P, E])

def extract_features_TVLDA(epochs, ratio=0.7):

    epochs.pick_types(eeg=True)
    X_event, X_nonevent, y = balance_data(epochs, ratio)
    w = 6
    freq = np.array([2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    fs = epochs.info["sfreq"]
    nCh = 8
    nSample = X_event.shape[2]
    widths = w*fs / (2*freq*np.pi)
    dt = 1 / fs

    cwtm_X_events = np.empty((X_event.shape[0], nCh, freq.shape[0], nSample))
    for i_epoch in range(X_event.shape[0]):
        for i_ch in range(nCh):
            coeffs, freqs = pywt.cwt(X_event[i_epoch, i_ch, :],
                                     scales=widths,
                                     wavelet='cmor',
                                     sampling_period=dt)
            cwtm_X_events[i_epoch, i_ch, :, :] = np.abs(coeffs)

    cwtm_X_nonevents = np.empty((X_nonevent.shape[0], nCh, freq.shape[0], nSample))
    for i_epoch in range(X_nonevent.shape[0]):
        for i_ch in range(nCh):
            coeffs, freqs = pywt.cwt(X_nonevent[i_epoch, i_ch, :],
                                     scales=widths,
                                     wavelet='cmor',
                                     sampling_period=dt)
            cwtm_X_nonevents[i_epoch, i_ch, :, :] = np.abs(coeffs)

    ## @leo use these: (nEpochs, nCh, nFreq, nTime, nSampleWithinChunk) (150, 8, 12, 10, 22)
    TVLDA_event = cwtm_X_events[:, :, :, :-6].reshape(cwtm_X_events.shape[0],
                                                       cwtm_X_events.shape[1],
                                                       12, 10, 22)
    ## @leo use these: (nEpochs, nCh, nFreq, nTime, nSampleWithinChunk) (150, 8, 12, 10, 22)
    TVLDA_nonevent = cwtm_X_nonevents[:, :, :, :-6].reshape(cwtm_X_nonevents.shape[0],
                                                            cwtm_X_nonevents.shape[1],
                                                            12, 10, 22)

    # --- Compute statistical features and append along the time dimension ---
    # For each epoch, channel and frequency, aggregate over original (nTime, nSampleWithinChunk)
    # to get 8 features. Then repeat each statistic 22 times to form a new slice.
    nEpochs, _, nFreq, nTime, nSample = TVLDA_event.shape
    nStat = 8  # number of stat features
    stat_features_event = np.empty((nEpochs, nCh, nFreq, nStat))
    stat_features_nonevent = np.empty((TVLDA_nonevent.shape[0], nCh, nFreq, nStat))
    
    for i in range(nEpochs):
        for j in range(nCh):
            for k in range(nFreq):
                # Flatten entire chunk (both time and sample dims) for statistical summary
                seg_event = TVLDA_event[i, j, k, :, :].flatten()
                stat_features_event[i, j, k, :] = extract_stat_features(seg_event)
                
    for i in range(TVLDA_nonevent.shape[0]):
        for j in range(nCh):
            for k in range(nFreq):
                seg_nonevent = TVLDA_nonevent[i, j, k, :, :].flatten()
                stat_features_nonevent[i, j, k, :] = extract_stat_features(seg_nonevent)
                
    # Expand stat features along a new time axis by repeating each statistic to match nSample (22)
    stat_features_event = np.repeat(stat_features_event[:, :, :, :, np.newaxis], nSample, axis=4)
    stat_features_nonevent = np.repeat(stat_features_nonevent[:, :, :, :, np.newaxis], nSample, axis=4)
    # stat_features_event has shape: (nEpochs, nCh, nFreq, 8, nSample)
    # Concatenate along the time axis (axis=3): new nTime becomes 10 + 8 = 18
    TVLDA_event_full = np.concatenate([TVLDA_event, stat_features_event], axis=3)
    TVLDA_nonevent_full = np.concatenate([TVLDA_nonevent, stat_features_nonevent], axis=3)

    # Update the bis versions accordingly
    TVLDA_event_bis_full = TVLDA_event_full.reshape(nEpochs, nCh * nFreq * (nTime + nStat), nSample)
    TVLDA_nonevent_bis_full = TVLDA_nonevent_full.reshape(TVLDA_nonevent_full.shape[0],
                                                            nCh * nFreq * (nTime + nStat), nSample)

    return {"event_raw": TVLDA_event_full,
            "nonevent_raw": TVLDA_nonevent_full,
            "event": TVLDA_event_bis_full,
            "nonevent": TVLDA_nonevent_bis_full}