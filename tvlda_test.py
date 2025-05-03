#%%
from feature_extraction import *
from preprocessing import *
from models.tvlda import TVLDA
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
filename = os.path.join('p300-speller','S1.mat')

epochs = get_epochs_from_file(filename)
features_TVLDA = extract_features_TVLDA(epochs)
#%%
positive_events = torch.tensor(features_TVLDA['event'])
negative_events = torch.tensor(features_TVLDA['nonevent'])
negative_events = negative_events[:positive_events.shape[0],...]
# Split positive and negative events separately (80% training, 20% testing)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

accs, precisions, recalls, f1s = [], [], [], []
cm_total = np.zeros((2, 2), dtype=int)

for train_idx, test_idx in kf.split(positive_events):
    positive_train = positive_events[train_idx]
    positive_test  = positive_events[test_idx]
    negative_train = negative_events[train_idx]
    negative_test  = negative_events[test_idx]

    tvlda_model = TVLDA()
    tvlda_model.fit(positive_train, negative_train)
    tvlda_model.fit_score(positive_train, 1)
    predictions_neg = tvlda_model.get_label(negative_test)
    tn = (predictions_neg == -1).sum().item()
    fp = (predictions_neg == 1).sum().item()

    predictions_pos = tvlda_model.get_label(positive_test)
    fn = (predictions_pos == -1).sum().item()
    tp = (predictions_pos == 1).sum().item()

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    accs.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1_score)

    cm_total += np.array([[tn, fp],
                          [fn, tp]])

# Average metrics over folds
accuracy = np.mean(accs)
precision = np.mean(precisions)
recall = np.mean(recalls)
f1 = np.mean(f1s)

# Use the aggregated confusion matrix for plotting
true_neg, false_pos = cm_total[0]
false_neg, true_pos = cm_total[1]
# Actual negatives: [true_neg, false_pos]
# Actual positives: [false_neg, true_pos]
cm = np.array([[true_neg, false_pos],
               [false_neg, true_pos]])

# Compute evaluation metrics
total = true_pos + true_neg + false_pos + false_neg
accuracy = (true_pos + true_neg) / total if total else 0
precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0
recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("Prediction metrics:")
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))

# Plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')

# Set ticks and labels
ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(2))
ax.set_xticklabels(['Pred Negative', 'Pred Positive'])
ax.set_yticklabels(['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Loop over data dimensions and add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.colorbar(im)
plt.show()
# %%
