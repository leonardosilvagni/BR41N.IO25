#%%
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
#%%
trigger_file_path = 'p300-speller/S1.mat'
S1_mat = scipy.io.loadmat(trigger_file_path)

S1_data = S1_mat['y']
S1_trigger = S1_mat['trig']
fs = int(S1_mat['fs'][0][0])
nSamples = S1_data.shape[0]

#%% visualize the data
ch = 0
import plotly.graph_objects as go
x = np.linspace(0, nSamples/fs, nSamples).reshape(-1)
fig= go.Figure(go.Scatter(x=x, y= S1_trigger.ravel(), name=f"Events", mode ="lines", fill='tozeroy'))  

fig.add_scatter(x=x, y=S1_data[:, ch]/S1_data[:, ch].max(), mode="lines", name =f"Ch{k+1}")
fig.update_layout(width=950, title="ECoG recordings", title_x=0.5, 
                  xaxis_title="s", yaxis_title="voltage?")    
fig.show()
# %%

target_ind = np.where(S1_trigger==1)[0] #  150
nontarget_ind = np.where(S1_trigger==-1)[0] # 1050
# %%
nTargets = target_ind.shape[0]
nNonTargets = nontarget_ind.shape[0]

preEpoch_s = 1
postEpoch_s = 1

preEpoch_ind = int(preEpoch_s * fs)
postEpoch_ind = int(postEpoch_s * fs)

epoch_ind = np.arange(-preEpoch_ind, postEpoch_ind)

epochsTarget = np.full((nTargets, len(epoch_ind), 8), np.nan)   # (targets, time, ch)

# extract target epochs
for iEpoch in range(nTargets):
    ind = target_ind[iEpoch] + epoch_ind
    # print(target_ind[iEpoch], ind[0], ind[-1])

    epochsTarget[iEpoch, :, :] = S1_data[ind, :]


epochsNonTarget = np.full((nNonTargets, len(epoch_ind), 8), np.nan)   # (targets, time, ch)

# extract target epochs
for iEpoch in range(nNonTargets):
    ind = nontarget_ind[iEpoch] + epoch_ind
    # print(target_ind[iEpoch], ind[0], ind[-1])

    epochsNonTarget[iEpoch, :, :] = S1_data[ind, :]

meanTarget = np.nanmean(epochsTarget, axis=0)
meanNonTarget = np.nanmean(epochsNonTarget, axis=0)

#%%
t = np.linspace(-preEpoch_s, postEpoch_s, len(epoch_ind))

#%%

plt.plot(t, meanTarget[:, 0], label='target')
plt.plot(t, meanNonTarget[:, 0], label='non-target')
# %%
for j in range(150):
    plt.plot(t, epochsTarget[j, :, 0], label='target')

plt.xlim(0, 1)
# %%
epochsTarget.shape
t.shape
# %%

# Create an 8x8 grid of subplots
fig = make_subplots(
    rows=8, cols=8,  # 8x8 grid
    shared_xaxes=True,  # Shared x-axis for all plots
    shared_yaxes=True,  # Shared y-axis for all plots
    vertical_spacing=0.02,  # Adjust spacing between subplots
    horizontal_spacing=0.02,  # Adjust spacing between subplots
    subplot_titles=[f"Electrode {electrodes[ch_n]}" for ch_n in range(64)]  # Titles for each subplot
)

# Iterate over the channels (ch_n = 0 to 63)
for ch_n in range(64):
    # Add traces for each subplot
    fig.add_trace(go.Scatter(
        x=t.ravel(), y=(meanError[:, ch_n] - meanError[0, ch_n]).ravel(),
        line=dict(width=1), mode="lines", opacity=0.7, showlegend=False
    ), row=(ch_n // 8) + 1, col=(ch_n % 8) + 1)  # Row and column for the subplot

    fig.add_trace(go.Scatter(
        x=t.ravel(), y=(meanEnd[:, ch_n] - meanEnd[0, ch_n]).ravel(),
        line=dict(width=1), mode="lines", opacity=0.7, showlegend=False
    ), row=(ch_n // 8) + 1, col=(ch_n % 8) + 1)

    fig.add_trace(go.Scatter(
        x=t.ravel(), y=(meanStart[:, ch_n] - meanStart[0, ch_n]).ravel(),
        line=dict(width=1), mode="lines", opacity=0.7, showlegend=False
    ), row=(ch_n // 8) + 1, col=(ch_n % 8) + 1)

    if ch_n == 0:
        fig.add_trace(go.Scatter(
            x=t.ravel(), y=(meanError[:, ch_n] - meanEnd[:, ch_n]).ravel(),
            line=dict(width=3, color='black'), mode="lines", name = 'error-correct', showlegend=True
        ), row=(ch_n // 8) + 1, col=(ch_n % 8) + 1)
    else:
        fig.add_trace(go.Scatter(
            x=t.ravel(), y=(meanError[:, ch_n] - meanEnd[:, ch_n]).ravel(),
            line=dict(width=3, color='black'), mode="lines", name = 'error-correct', showlegend=False
        ), row=(ch_n // 8) + 1, col=(ch_n % 8) + 1)

# Update layout (size of the figure, axis labels, etc.)
fig.update_layout(
    width=2000, height=2000,
    title="Electrode Data for All Channels",
    showlegend=True,  # Hide legend if you want to reduce clutter
    title_x=0.5,  # Center the title
    xaxis_title="Time to Event (s)",
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
    ),
    # yaxis=dict(range=[-9.5, 9.5]),
)
fig.show()