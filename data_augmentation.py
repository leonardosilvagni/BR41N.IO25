from preprocessing import * 
import numpy as np

def balance_data(epochs, ratio=0.6):
    # Select only EEG channels
    epochs.pick_types(eeg=True)
    times = epochs.times
    y_orig = epochs.events[:, -1]

    # Get data for events and non-events
    X_event = epochs['event'].get_data()*1e3
    X_nonevent = epochs['non_event'].get_data()*1e3
    n_event = X_event.shape[0]
    n_nonevent = X_nonevent.shape[0]

    # Generate synthetic event epochs by averaging two randomly chosen event epochs.
    # The number of synthetic epochs depends on the ratio
    synthetic_needed = int(ratio * (n_nonevent - n_event))
    synthetic_epochs = []
    for _ in range(synthetic_needed):
        # Pick two distinct event indices at random
        i1, i2 = np.random.choice(n_event, 2, replace=False)
        synthetic_epoch = (X_event[i1] + X_event[i2]) / 2.0
        synthetic_epochs.append(synthetic_epoch)
    synthetic_epochs = np.array(synthetic_epochs)
    
    if ratio == 0:
        y_event = np.ones(X_event.shape[0])
        y_nonevent = -np.ones(X_nonevent.shape[0])
        
        # Combine the balanced classes
        X_data = np.concatenate([X_event, X_nonevent], axis=0)
        y_data = np.concatenate([y_event, y_nonevent])
        return X_data, y_data, times    # Augment the original events with the synthetic epochs
    X_event_augmented = np.concatenate([X_event, synthetic_epochs], axis=0)
    
    # Downsample the non-event epochs so that we have the same number of non-event epochs as events
    #target_count = X_event_augmented.shape[0]
    # Ensure we sample without replacement
    #sampled_indices = np.random.choice(n_nonevent, target_count, replace=False)
    X_nonevent_balanced = X_nonevent#[sampled_indices]
    
    # Optionally, build labels for your dataset (assuming label 1 for events and 0 for non-events)
    y_event = np.ones(X_event_augmented.shape[0])
    y_nonevent = -np.ones(X_nonevent_balanced.shape[0])
    
    # Combine the balanced classes
    X_balanced = np.concatenate([X_event_augmented, X_nonevent_balanced], axis=0)
    y_balanced = np.concatenate([y_event, y_nonevent])
    
    return X_balanced, y_balanced, times