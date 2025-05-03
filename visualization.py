from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
def visualize_epochs(epochs):

    # Create the time axis (in seconds)
    t = epochs.times  
    ch_names = ["Fz","Cz","P3","Pz","P4","PO7","PO8","Oz"] 

    meanEvent = epochs['event'].average().get_data()  # Mean across epochs for target events
    meanNonEvent = epochs['non_event'].average().get_data() # Mean across epochs for non-target events

    # Create an 8x8 grid of subplots
    fig = make_subplots(
        rows=4, cols=2,  # 8x8 grid
        shared_xaxes=True,  # Shared x-axis for all plots
        shared_yaxes=True,  # Shared y-axis for all plots
        vertical_spacing=0.05,  # Adjust spacing between subplots
        horizontal_spacing=0.02,  # Adjust spacing between subplots
        subplot_titles=[f"{ch_names[ch_n]}" for ch_n in range(8)]  # Titles for each subplot
    )

    # Iterate over the channels (ch_n = 0 to 63)
    for ch_n in range(8):
        # print(ch_n)
        # print(ch_n % 4) 
        # print(ch_n % 2)
        # Add traces for each subplot
        fig.add_trace(go.Scatter(
            x=t.ravel(), y=(meanEvent[ch_n, :] - np.mean(meanNonEvent[ch_n, :])).ravel(),
            line=dict(width=1, color = "blue"), mode="lines", opacity=0.7, showlegend=False
        ), row=(ch_n // 2) + 1, col=(ch_n % 2) + 1)  # Row and column for the subplot

        fig.add_trace(go.Scatter(
            x=t.ravel(), y=(meanNonEvent[ch_n, :] - np.mean(meanNonEvent[ch_n, :])).ravel(),
            line=dict(width=1, color = "red"), mode="lines", opacity=0.7, showlegend=False
        ), row=(ch_n // 2) + 1, col=(ch_n % 2) + 1)  # Row and column for the subplot


    # Update layout (size of the figure, axis labels, etc.)
    fig.update_layout(
        width=800, height=800,
        title="ERP for All Channels",
        # showlegend=True,  # Hide legend if you want to reduce clutter
        title_x=0.5,  # Center the title
        # xaxis_title="Time to Event (s)",
        yaxis_title="Voltage",
        legend=dict(
            x=1,  # Position the legend on the right
            y=1,
            traceorder="normal",  # Order traces in the legend
            orientation="v",  # Vertical orientation of the legend
            font=dict(size=12),  # Font size of the legend labels
            bgcolor="rgba(255, 255, 255, 0.7)",  # Background color of the legend
            bordercolor="black",  # Border color of the legend
            borderwidth=2  # Border width of the legend
        )
    )
    fig.update_xaxes(title_text="Time to Event (s)", row=4, col=1)
    fig.update_xaxes(title_text="Time to Event (s)", row=4, col=2)

    fig.show()

def plot_metrics(conf_matrix, auc):
    """
    Plot the confusion matrix and AUC score.

    Args:
        conf_matrix (np.ndarray): Confusion matrix.
        auc (float): Area under the ROC curve.
    """
    print("Aggregated Confusion Matrix:")
    print(conf_matrix)
    print("AUC: {:.4f}".format(auc))

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, cmap='Blues')

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Pred Negative', 'Pred Positive'])
    ax.set_yticklabels(['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Aggregated Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="red")

    plt.colorbar(im)
    plt.show()
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    tp = conf_matrix[1, 1]

    total = conf_matrix.sum()
    accuracy = (tp + tn) / total

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 Score: {:.4f}".format(f1_score))