#%%
from preprocessing import *

# Some standard pythonic imports
import warnings
warnings.filterwarnings('ignore')
import os,numpy as np,pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt
import pywt
from data_augmentation import balance_data

def extract_features_TVLDA(epochs,ratio=0.7):

    epochs.pick_types(eeg=True)
    # X = epochs.get_data() * 1e6
    times = epochs.times
    y = epochs.events[:, -1]

    #X_event = epochs['event'].get_data()
    #X_nonevent = epochs['non_event'].get_data()
    X_event, X_nonevent, y = balance_data(epochs, ratio)
    # from scipy impywtport signal
    # from scipy.signal import cwt, morlet2
    w = 6
    freq = np.array([2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    fs = epochs.info["sfreq"]
    nCh = 8
    nSample = X_event.shape[2]
    widths = w*fs / (2*freq*np.pi)
    dt = 1 / fs

    cwtm_X_events = np.empty((X_event.shape[0], nCh, freq.shape[0], nSample))#, dtype=complex)

    for i_epoch in range(X_event.shape[0]):
        for i_ch in range(nCh):
            coeffs, freqs = pywt.cwt(X_event[i_epoch, i_ch, :], scales=widths,
                                    wavelet='cmor', sampling_period=dt)
            cwtm_X_events[i_epoch, i_ch, :, :] = np.abs(coeffs)

    cwtm_X_nonevents = np.empty((X_nonevent.shape[0], nCh, freq.shape[0], nSample))#, dtype=complex)

    for i_epoch in range(X_nonevent.shape[0]):
        for i_ch in range(nCh):
            coeffs, freqs = pywt.cwt(X_nonevent[i_epoch, i_ch, :], scales=widths,
                                    wavelet='cmor', sampling_period=dt)
            cwtm_X_nonevents[i_epoch, i_ch, :, :] = np.abs(coeffs)


    ## @leo use these: (nEpochs, nCh, nFreq, nTime, nSampleWithinChunk) (150, 8, 12, 10, 22)
    TVLDA_event = cwtm_X_events[:,:,:, :-6].reshape(cwtm_X_events.shape[0], cwtm_X_events.shape[1], 12, 10, 22)
    TVLDA_event_bis = TVLDA_event.reshape(cwtm_X_events.shape[0], cwtm_X_events.shape[1]* 12* 10, 22)




    mean_cwtm_X_nonevents = np.mean(np.abs(cwtm_X_nonevents), axis=0)


    ## @leo use these: (nEpochs, nCh, nFreq, nTime, nSampleWithinChunk) (150, 8, 12, 10, 22)
    TVLDA_nonevent = cwtm_X_nonevents[:,:,:, :-6].reshape(cwtm_X_nonevents.shape[0], cwtm_X_nonevents.shape[1], 12, 10, 22)
    print(TVLDA_nonevent.shape)
    TVLDA_nonevent_bis = TVLDA_nonevent.reshape(cwtm_X_nonevents.shape[0], cwtm_X_nonevents.shape[1]* 12* 10, 22)

    return {"event_raw": TVLDA_event, "nonevent_raw": TVLDA_nonevent, "event": TVLDA_event_bis, "nonevent": TVLDA_nonevent_bis}