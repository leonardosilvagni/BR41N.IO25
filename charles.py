#%%
import scipy.io
import numpy as np

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

plt.plot(meanTarget[:, 0], label='target')
plt.plot(meanNonTarget[:, 0], label='non-target')
# %%
plt.plot()