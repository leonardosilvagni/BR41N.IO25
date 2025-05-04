from preprocessing import * 
import numpy as np
import torch
def balance_data(X, y, ratio=0.6, noise=False):
    # Select event and non-event data based on labels
    if ratio == 0:
        return X, y
    X_event = X[y == 1]
    X_nonevent = X[(y == -1) | (y == 0)]
    n_event = X_event.shape[0]
    n_nonevent = X_nonevent.shape[0]

    # Generate synthetic event samples by averaging two randomly chosen event samples.
    synthetic_needed = int(ratio * (n_nonevent - n_event))
    synthetic_epochs = []
    for _ in range(synthetic_needed):
        # Pick two distinct event indices at random
        i1, i2 = np.random.choice(n_event, 2, replace=False)
        synthetic_epoch = (X_event[i1] + X_event[i2]) / 2.0
        synthetic_epochs.append(synthetic_epoch)
    synthetic_epochs = np.array(synthetic_epochs) if synthetic_epochs else np.empty((0,) + X_event.shape[1:])

    # If no synthetic data is needed, simply combine existing event and non-event data.


    # Augment the event samples with the synthetic epochs
    X_event_augmented = np.concatenate([X_event, synthetic_epochs], axis=0)

    # Non-event epochs remain unchanged
    X_nonevent_balanced = X_nonevent

    # Create labels: 1 for events and -1 for non-events
    y_event_augmented = np.ones(X_event_augmented.shape[0])
    y_nonevent = -np.ones(X_nonevent_balanced.shape[0])

    # Combine the balanced classes
    X_balanced = torch.tensor(np.concatenate([X_event_augmented, X_nonevent_balanced], axis=0))
    y_balanced = torch.tensor(np.concatenate([y_event_augmented, y_nonevent]))
    if noise:
        x_aug = X_balanced + noise * torch.randn_like(X_balanced)
        print("Added noise to the data")
        return torch.cat([X_balanced, x_aug]), torch.cat([y_balanced, y_balanced])   # if your dataloader can handle batching
    return X_balanced, y_balanced
