#%%
from preprocessing import *

# Some standard pythonic imports
import warnings
warnings.filterwarnings('ignore')
import os,numpy as np,pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt

subject_ids = [1, 2, 3, 4, 5]  # or however many subjects you have
folder = 'p300-speller'

sid = 1
filepath = os.path.join(folder, f"S{sid}.mat")

print(f"Loading subject {sid} from {filepath}")
epochs_S1 = get_epochs_from_file(filepath)
epochs_S1.pick_types(eeg=True)
# X = epochs_S1.get_data() * 1e6
times = epochs_S1.times
y = epochs_S1.events[:, -1]

X_event = epochs_S1['event'].get_data()
X_nonevent = epochs_S1['non_event'].get_data()

#%%
import pywt
# from scipy impywtport signal
# from scipy.signal import cwt, morlet2
w = 6
freq = np.array([2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50])
fs = epochs_S1.info["sfreq"]
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

# %%
mean_cwtm_X_events = np.mean(np.abs(cwtm_X_events), axis=0)

plt.figure(figsize=(12, 10))
plt.imshow(mean_cwtm_X_events[0, :, :], aspect='auto', extent=[times[0], times[-1], freq[0], freq[-1]], cmap='jet')
plt.title("Mean CWT of Event")
plt.colorbar(label='Magnitude')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()

#%%
print(times[0], times[-2])
print((times[-2]-times[0])/12)

## @leo use these: (nEpochs, nCh, nFreq, nTime, nSampleWithinChunk) (150, 8, 12, 10, 22)
TVLDA_event = cwtm_X_events[:,:,:, :-6].reshape(cwtm_X_events.shape[0], cwtm_X_events.shape[1], 12, 10, 22)
print(TVLDA_event.shape)
TVLDA_event_bis = TVLDA_event.reshape(cwtm_X_events.shape[0], cwtm_X_events.shape[1]* 12* 10, 22)
print(TVLDA_event_bis.shape)
mean_TVLDA_event = np.mean(TVLDA_event, axis=0)
mean_TVLDA_event.shape
mean_TVLDA_event_chuncked = np.mean(mean_TVLDA_event, axis=3)

plt.figure(figsize=(12, 10))
plt.imshow(mean_TVLDA_event_chuncked[0, :, :], aspect='auto', extent=[times[0], times[-1], freq[0], freq[-1]], cmap='jet')
plt.title("Mean CWT of Event TLVDA")
plt.colorbar(label='Magnitude')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()

#%%
## only get the data starting from 0.0s to 0.8s 
print(times[25], times[-1])
reshaped = cwtm_X_events[:,:,:,25:-1].reshape(150, 8, 12, 10, 20)  # 10 chunks of 20
reduced = reshaped.mean(axis=-1)
reduced_mean = reduced.mean(axis=0)
reduced_mean.shape

plt.figure(figsize=(12, 10))
plt.imshow(reduced_mean[0, :, :], aspect='auto', extent=[-0.1, 0.8, freq[0], freq[-1]], cmap='jet')
plt.title("Mean CWT of Event")
plt.colorbar(label='Magnitude')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()

#%%x

mean_cwtm_X_nonevents = np.mean(np.abs(cwtm_X_nonevents), axis=0)

plt.figure(figsize=(12, 10))
plt.imshow(mean_cwtm_X_nonevents[0, :, :], aspect='auto', extent=[times[0], times[-1], freq[0], freq[-1]], cmap='jet')
plt.title("Mean CWT of Non-Event")
plt.colorbar(label='Magnitude')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()

#%%
print(times[0], times[-2])
print((times[-2]-times[0])/12)

## @leo use these: (nEpochs, nCh, nFreq, nTime, nSampleWithinChunk) (150, 8, 12, 10, 22)
TVLDA_nonevent = cwtm_X_nonevents[:,:,:, :-6].reshape(cwtm_X_nonevents.shape[0], cwtm_X_nonevents.shape[1], 12, 10, 22)
print(TVLDA_nonevent.shape)
TVLDA_nonevent_bis = TVLDA_nonevent.reshape(cwtm_X_nonevents.shape[0], cwtm_X_nonevents.shape[1]* 12* 10, 22)
print(TVLDA_nonevent_bis.shape)
mean_TVLDA_nonevent = np.mean(TVLDA_nonevent, axis=0)
mean_TVLDA_nonevent.shape
mean_TVLDA_nonevent_chuncked = np.mean(mean_TVLDA_nonevent, axis=3)

plt.figure(figsize=(12, 10))
plt.imshow(mean_TVLDA_nonevent_chuncked[0, :, :], aspect='auto', extent=[times[0], times[-1], freq[0], freq[-1]], cmap='jet')
plt.title("Mean CWT of Non-Event TLVDA")
plt.colorbar(label='Magnitude')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()

#%%
## only get the data starting from 0.0s to 0.8s 
print(times[25], times[-1])
reshaped = cwtm_X_nonevents[:,:,:,25:-1].reshape(cwtm_X_nonevents.shape[0], 8, 12, 10, 20)  # 10 chunks of 20
reduced = reshaped.mean(axis=-1)
reduced_mean = reduced.mean(axis=0)
reduced_mean.shape

plt.figure(figsize=(12, 10))
plt.imshow(reduced_mean[0, :, :], aspect='auto', extent=[-0.1, 0.8, freq[0], freq[-1]], cmap='jet')
plt.title("Mean CWT of Non-Event")
plt.colorbar(label='Magnitude')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()
# %%