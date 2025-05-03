# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import mne
from collections import OrderedDict
import scipy.io as sio
mat_data = {}
# Load the MAT file (change idx or filename as needed)
folder_path = "p300-speller"
for idx in range(1, 2):
    file_path = os.path.join(folder_path, f"S{idx}.mat")
    print(file_path)
    mat_data[f"S{idx}"] = sio.loadmat(file_path)

# Keep only S1 for now
mat_content = mat_data["S1"]
# Extract the data from the dictionary
# The file contains keys: __header__, __version__, __globals__, 'fs', 'trig', 'y'
fs = mat_content['fs'][0][0]  # sampling frequency (assumed to be stored as a 2D array scalar)
y = mat_content['y']        # data for 8 channels
#%%
# It is assumed that y has shape (8, n_samples)
n_samples,n_channels = y.shape
print(f"Number of channels: {n_channels}, Number of samples: {n_samples}")
#%%
time = np.arange(n_samples) / fs   # create a time vector in seconds

# Plotting each channel
plt.figure(figsize=(12, 10))
for ch in range(n_channels):
    plt.subplot(n_channels, 1, ch+1)
    plt.plot(time, 100*mat_content["trig"], color="red", linestyle="--", linewidth=2)
    plt.plot(time, y[:, ch])
    plt.title(f"Channel {ch+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.show()
# %%
import scipy.signal

# Compute and plot the power spectral density for each channel
plt.figure(figsize=(12, 10))
for ch in range(n_channels):
    # Compute the power spectral density using periodogram
    f, psd = scipy.signal.periodogram(y[:, ch], fs)
    
    # Plot the PSD; semilogy scale to better visualize peaks (line interference typically appears around 50 or 60 Hz)
    plt.subplot(n_channels, 1, ch + 1)
    plt.semilogy(f, psd)
    plt.xlim(0, 100)  # Focus on frequencies up to 100 Hz to inspect line interference
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(f"Power Spectrum for Channel {ch + 1}")
    plt.grid(True)

plt.tight_layout()
plt.show()

# %%
# Prepare EEG info and create a Raw object (MNE expects data shape as channels x samples)
ch_names = ["Fz","Cz","P3","Pz","P4","PO7","PO8","Oz"]  # Channel names
ch_types = ["eeg"] * n_channels
info = mne.create_info(ch_names, fs, ch_types)
raw = mne.io.RawArray(y.T, info)

# Prepare the trigger channel (flattened to 1D); here we assume its values are 1 (event) or -1 (non event)
trig = mat_content["trig"].ravel()

# Identify the indices where trigger is either 1 or -1
event_idxs = np.where(np.abs(trig) == 1)[0]
events = np.column_stack((event_idxs, np.zeros(event_idxs.size, dtype=int), trig[event_idxs].astype(int)))

# Define event IDs according to the trigger values
event_id = {"event": 1, "non_event": -1}

# Create epochs: here an epoch spans from -0.2 to 0.5 seconds around each event
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.1, tmax=0.8, baseline=(None, 0), preload=True)

print(epochs)

#%%
# Compute the average for each event type
evoked_event = epochs["event"].average()
evoked_non_event = epochs["non_event"].average()

# Plot the averaged epochs for "event"
fig_event = evoked_event.plot(spatial_colors=True, time_unit="s")
# Plot the averaged epochs for "non_event"
fig_non_event = evoked_non_event.plot(spatial_colors=True, time_unit="s")

# %%
