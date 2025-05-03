import numpy as np
import matplotlib.pyplot as plt
import os
import mne
from collections import OrderedDict
import scipy.io as sio

def get_epochs_from_file(file_path):
    """
    Load EEG data from a .mat file and create MNE epochs.
    """
    # Keep only S1 for now
    mat_content = sio.loadmat(file_path)
    # Extract the data from the dictionary
    # The file contains keys: __header__, __version__, __globals__, 'fs', 'trig', 'y'
    fs = mat_content['fs'][0][0]  # sampling frequency (assumed to be stored as a 2D array scalar)
    y = mat_content['y']        # data for 8 channels

    # It is assumed that y has shape (8, n_samples)
    n_samples,n_channels = y.shape
    print(f"Number of channels: {n_channels}, Number of samples: {n_samples}")

    time = np.arange(n_samples) / fs   # create a time vector in seconds



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
    return epochs
