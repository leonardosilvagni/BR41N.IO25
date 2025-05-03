from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

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