# %%
# Date: Dec-15-2025
# Author: Simon Chan Tack
# File: MainSingleEng_FD002_ F004_OptHypTune.py
# Code execute entire pipeline of the Multi-Dimension Transformer architecture
# Preforms Optimal Hyperparameter tuning on engine FD002 and FD004 dataset
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from Encoder_Layers import *
from CommonFunctions import *
import plotly.express as px
import matplotlib.pyplot as plt
from keras import models, layers
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error as mse


# %%
# Prepare to set up and import data
column_names = ['engine', 'time', 'op_setting_1', 'op_setting_2', 
                'op_setting_3'] + [f'sm_{i}' for i in range(1, 22)]


# assign names to columns , save in dict_list 
Sensor_dictionary={}
dict_list=[ "(Fan inlet temperature) (◦R)",
"(LPC outlet temperature) (◦R)",
"(HPC outlet temperature) (◦R)",
"(LPT outlet temperature) (◦R)",
"(Fan inlet Pressure) (psia)",
"(bypass-duct pressure) (psia)",
"(HPC outlet pressure) (psia)",
"(Physical fan speed) (rpm)",
"(Physical core speed) (rpm)",
"(Engine pressure ratio(P50/P2)",
"(HPC outlet Static pressure) (psia)",
"(Ratio of fuel flow to Ps30) (pps/psia)",
"(Corrected fan speed) (rpm)",
"(Corrected core speed) (rpm)",
"(Bypass Ratio) ",
"(Burner fuel-air ratio)",
"(Bleed Enthalpy)",
"(Required fan speed)",
"(Required fan conversion speed)",
"(High-pressure turbines Cool air flow)",
"(Low-pressure turbines Cool air flow)" ]

i=1
for x in dict_list :
    Sensor_dictionary[f'sm_{i}']=x
    i+=1
Sensor_dictionary



# %%
# Load Test Data
# this is RUL of each engine on test set. 100 row 
import pandas as pd
eng_type = 'FD002'
data_test = pd.read_csv('Data/test_' + eng_type +'.txt' , sep = ' ' , 
                       header = None, names = column_names , index_col = False)

y_test = pd.read_csv('Data/RUL_' + eng_type + '.txt' , header=None , names=['RUL'] )

# Dimension of Test Data and Target
data_test.shape , y_test.shape

data_test


# %%
# Load Training Data
data_train = pd.read_csv('Data/train_' + eng_type + '.txt' , sep = ' ' , header=None ,
                          names=column_names , index_col=False )
# Dimension of Training Data
data_train.shape



# %%
# Data inspection - Check for missing data, unique values, and data types
df_info = pd.concat ( [data_train.isna().sum() , data_train.nunique() , data_train.dtypes] , axis = 1  )
df_info.columns = ['missing value' , 'number unique value' , 'dtype']
df_info



# %%
# Make copy of original dataset , assing new name for dataframes
df = data_train.copy()
df_test = data_test.copy()


# Identify any columns that do not show any activity or relatively constant values
# These columns would not be good features to determine chnages in equipment behavior
def constant_feature(df):
    constant_feature = []
    for col in df.columns:
            if abs(df[col].std() < 0.02):     
                constant_feature.append(col)
    
    return constant_feature

print(constant_feature(df))

# Remove columns whoes values are relatively constant
df.drop(columns=constant_feature(df)  , inplace = True)
df_test.drop(columns=constant_feature(df) , inplace = True)

df.columns


# %%
# Defining RUL for training dataset
# RUL is rest useful life for each engines instant 
# for engine 1 , max time will be true RUL , we can subtract every time from max time 
# it gives us rul for each engins' states
# it is grouped data by engine , and for every engine , take time's columns , and take max value of time
# assign max times of each engine for all of engine's , then subtract it by its time , result is rul
def create_rul(df):
    df['rul'] = df.groupby('engine')['time'].transform('max') - df['time']
    return df

create_rul(df)


# %%
# Visualize how the groups are created based on engine labels
groups = df.groupby('engine')['rul']
for group, grouped_df in groups:
    print(f'Group:{group}')
    print(f'Length of group {group}: {len(grouped_df)}')
    print(grouped_df)
    print()

# %% 
# Perform clustering of eninges based on their operational settings
op_condit_df = df[['op_setting_1', 'op_setting_2', 'op_setting_3']]

# Use K-means to find clusters
kmeans = KMeans(n_clusters = 6)  # This is the KMeans model
kmeans.fit(op_condit_df)
cluster_labels = kmeans.labels_
cluster_labels_lst = cluster_labels.tolist()
centroids = kmeans.cluster_centers_

# Assign cluster labels to data
df.insert(1, 'OperCluster', cluster_labels_lst)
df


# %%
# 3-D Plot of KMean Cluster
colors = px.colors.qualitative.Plotly  

# Create 3D scatter plot
fig = go.Figure()

# Plot each cluster
for cluster in range(6):
    cluster_df = df[df['OperCluster'] == cluster]
    fig.add_trace(go.Scatter3d(
        x=cluster_df['op_setting_1'],
        y=cluster_df['op_setting_2'],
        z=cluster_df['op_setting_3'],
        mode='markers',
        marker=dict(
            size=5,
            color=colors[cluster],
            symbol='circle',      # "o" marker shape
            line=dict(color='black', width=2)),  # outline to make visible
        name=f'Cluster {cluster}',
        showlegend=True
    ))

# Layout
fig.update_layout(
    title={
        'text': '3D K-Means Clustering of Operational Conditions',
        'x': 0.5,          # Centers title horizontally (0 = left, 1 = right)
        'xanchor': 'center',
        'yanchor': 'top'
    },    
    scene=dict(
        xaxis_title='Oper Setting 1',
        yaxis_title='Oper Setting 2',
        zaxis_title='Oper Setting 3'
    ),
    legend=dict(x=0, y=1,
        bordercolor='lightgray',
        borderwidth=1),
    margin=dict(l=100, r=100, b=50, t=100)    
)

fig.show()

# %%
# Save the trained model to a file
import joblib
joblib.dump(kmeans, "kmeans_model_F004.pkl")
print("KMeans model saved as kmeans_model.pkl")

# Load saved model
kmeans_loaded = joblib.load("kmeans_model_F004.pkl")

# Prepare test dataset (must have same features and scaling as training data)
op_condit_test_df = df_test[['op_setting_1', 'op_setting_2', 'op_setting_3']]

# Predict cluster assignments for the test data
test_cluster_labels = kmeans_loaded.predict(op_condit_test_df)

# Optionally, add the cluster assignments back into the test DataFrame
df_test.insert(1, 'OperCluster', test_cluster_labels)

df_test.head()


# %%
# Functions to prepare normalization of data based on cluster assignments
def parameters_form(df, cluster_col):
    """
    Compute per-cluster mean and standard deviation for each feature in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing features + cluster column
        cluster_col (str): Name of the column containing cluster labels

    Returns:
        parameters_mean_df (pd.DataFrame): cluster-wise mean (index = cluster, columns = features)
        parameters_std_df (pd.DataFrame): cluster-wise std (index = cluster, columns = features)
    """
    feature_cols = df.columns.drop(cluster_col)

    # Group by cluster and calculate mean/std for each feature
    parameters_mean_df = df.groupby(cluster_col)[feature_cols].mean()
    parameters_std_df = df.groupby(cluster_col)[feature_cols].std(ddof=0)

    return parameters_mean_df, parameters_std_df


def normalize_regime(df, cluster_col, parameters_mean_df, parameters_std_df):
    """
    Normalize each row of DataFrame based on its cluster assignment.

    Parameters:
        df (pd.DataFrame): DataFrame containing features + cluster column
        cluster_col (str): Name of the column containing cluster labels
        parameters_mean_df (pd.DataFrame): cluster-wise mean (from parameters_form)
        parameters_std_df (pd.DataFrame): cluster-wise std (from parameters_form)

    Returns:
        pd.DataFrame: Normalized DataFrame with same shape as input (cluster column preserved)
    """
    feature_cols = df.columns.drop(cluster_col)
    norm_df = df.copy()

    for cluster_id, group in df.groupby(cluster_col):
        cluster_mean = parameters_mean_df.loc[cluster_id]
        cluster_std = parameters_std_df.loc[cluster_id].replace(0, 1)  # avoid div by 0

        norm_df.loc[group.index, feature_cols] = (group[feature_cols] - cluster_mean) / cluster_std

    return norm_df

# %%
# For Training Dataset
# Create df_sub that has cluster label and sensor data
# This prepares a data frame to normalize sensor data based on cluster assignments
df_sub = df.drop(['engine', 'time', 'op_setting_1', 
                  'op_setting_2', 'op_setting_3', 'rul'], axis = 1)

# Noralize data based on cluster group
# Step 1: Compute per-cluster statistics
parameters_mean_df, parameters_std_df = parameters_form(df_sub, cluster_col="OperCluster")

# Step 2: Normalize per cluster
normalized_df = normalize_regime(
    df_sub,
    cluster_col="OperCluster",
    parameters_mean_df = parameters_mean_df,
    parameters_std_df = parameters_std_df
)

# Put back engine assignment with normalized data
norm_df = pd.concat([df['engine'], normalized_df, df['rul']], axis = 1)


# %%
# For Testing Dataset
# Create df_sub that has cluster label and sensor data
# This prepares a data frame to normalize sensor data based on cluster assignments
df_test_sub = df_test.drop(['engine', 'time', 'op_setting_1', 
                       'op_setting_2', 'op_setting_3'], axis = 1)

# # Noralize data based on cluster group
# Step 1: Compute per-cluster statistics
parameters_mean_df_test, parameters_std_df_test = parameters_form(df_test_sub, cluster_col="OperCluster")

# Step 2: Normalize per cluster
normalized_df_test = normalize_regime(
    df_test_sub,
    cluster_col="OperCluster",
    parameters_mean_df = parameters_mean_df,
    parameters_std_df = parameters_std_df
)

# Put back engine assignment with normalized data
norm_df_test = pd.concat([df_test['engine'], normalized_df_test], axis = 1)


# %%
# Compute correlation between sensor data and mask top section for correlation matrix for better readability
df_corr = norm_df.corr()
mask = np.tril(np.ones(df_corr.shape),k = -1).astype(bool)
df_corr = df_corr.where(mask)
df_corr

# %%
# Plot corrleation matrix using heat map
plt.figure(figsize = (12,5))
plt.title('correlation')
sns.heatmap(df_corr , annot=True , fmt = '0.2f' , cmap='Blues')

# %%
# Zone into highly correlated features 
plt.figure(figsize = (12,5))
mask = df_corr.where( abs(df_corr) > 0.9 ).isna()
sns.heatmap(df_corr , annot=True , fmt = '0.2f' , linewidths=0.1 , mask = mask , cmap='Blues')

# %%
# Function to detect features (sensors) with more than 95% correlation
high_corr = []
for col in df_corr.columns:
    for row in df_corr.index:
        if abs(df_corr.loc[col , row]) > 0.93 :
            high_corr.append((col , row))
high_corr


# %%
col_to_drop = []
if len(high_corr) > 0:
    for i in range(len(high_corr)):
        corr_cols = high_corr[i]
        if (abs(df_corr.loc['rul', corr_cols[0]]) > abs(df_corr.loc['rul', corr_cols[1]])):
            col_to_drop.append(corr_cols[1])
        else:
            col_to_drop.append(corr_cols[0])

# Add additional columns to drop due to haviing zero correlation values
col_to_drop = ['sm_1', 'sm_5', 'sm_18', 'sm_19'] + col_to_drop 


# %%
# Function for Sensor visualization - plot signals from sensors
def plot_signal(df , signal_name , Sensor_dictionary):
    figure = plt.figure(figsize=(10,10))

    for engine in df['engine'].unique():  # Do for every 10th engine
        if (engine % 10 ==0 ):
            #print(engine)
            rolling_window = df[ df['engine']==engine ].rolling(10).mean()
            sns.lineplot( data = rolling_window , x = 'rul' , y =signal_name  , label =engine)
    
    plt.tight_layout(), plt.xlim(250 , 0)
    plt.title(signal_name + ': ' + Sensor_dictionary[signal_name] + ' vs Remainded Usefull Life (RUL)')
    plt.xlabel('Remainded Usefull Life (RUL)') , plt.ylabel(Sensor_dictionary[signal_name])
    plt.show()


# Plot sensor data for engines
for i in range (1,22):
    try:
        plot_signal(norm_df , 'sm_'+str(i)  , Sensor_dictionary)
    except:
        pass

# %%
# The feature has very high correlation , no need for both of them , we can drop one of them
# Also drop the features that have zero correlation 
# Do it for both Training and Testing dataset
norm_df.drop(columns = col_to_drop , inplace = True)  
norm_df_test.drop(columns = col_to_drop , inplace = True)
print(norm_df.columns)
print(norm_df_test.columns)

# %%
# Identify the features to be used in the models
features = norm_df.columns[2:-1]   # drop  engine , time , rul of  dataset
features 


# %%
# Piecewise function to prepare y (target) for training data
# RUL is the target variable, and it is the last column in the dataframe
norm_df['rul'] = np.where(norm_df['rul'] >= 125, 125, norm_df['rul'])  # Ensure RUL is non-negative

# Prepare blind test set - extract only features that will be use for testing
y_test['RUL'] = np.where(y_test['RUL'] >= 125, 125, y_test['RUL'])  # Ensure RUL is non-negative
y_test                       # y_test is blind test's target 


X = norm_df.copy()
X_test = norm_df_test.copy()

# %%
#####################################################################################################################
window_type = 'sliding'

# Sliding window function to prepare training data
# Parameters to prepare training data
if window_type == 'sliding':
    window_size = 40    # Each batch will have a sequence of window_size series
    X_train_sw, y_train_sw = create_training_sequences_sw(X, features, window_size)


    # Sliding window function to prepare testing data
    # Define how to shift window and number of batches
    shift = 1
    num_of_batches = 1

    X_testf = create_testing_sequences_sw(X_test, features, window_size, num_of_batches)
    y_test = np.squeeze(y_test).to_numpy()
    X_testf.shape



#########################################
# Training set
# from sklearn.model_selection import train_test_split
# X_train, X_val , y_train, y_val  = train_test_split(X_train_sw, y_train_sw, test_size=0.2 , random_state=42)



# %%
# Paper
# It accepts a list of arrays of varying sequence lengths and targets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


max_len = pd.unique(data_train['engine']).max() 

class supDataset(Dataset):
  def __init__(self, data_list, targets):
    self.data_list = data_list
    self.targets = targets

  # Returns len of dataset
  def __len__(self):
    return len(self.data_list)

  # Takes indices of data len, returns a dictionary of tensors
  def __getitem__(self, idx):
    X = self.data_list[idx]
    y = self.targets[idx]
    # return X, y
    # return torch.tensor(X, dtype=torch.float),  torch.tensor(y, dtype=torch.int64)
    return torch.tensor(X, dtype=torch.float), y

 



# %%
# Training function
import joblib
# Create device object to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda:0
# device = torch.device("cpu")
print(device)


# %%
# -----------------------------
# PatchTST blocks
# -----------------------------
import os
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------
# Patch Embedding (time --> tokens)
# ---------------------------------
class PatchEmbedding(nn.Module):
    """
    Turn a (B*C, L) series into a sequence of patch tokens (B*C, N, d_model).
    Each token is a linear projection of a length-P patch.
    """
    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):  # x: (B*C, L)
        # Dimension error check
        L = x.shape[1]
        if L < self.patch_len:
            raise ValueError(f"Lookback L={L} < patch_len={self.patch_len}. Increase lookback or reduce patch_len.")
        
        # N = floor((L - P)/stride) + 1
        n_patches = 1 + (L - self.patch_len) // self.stride
        if n_patches <= 0:
            raise ValueError("No patches would be created; check patch_len/stride vs lookback.")
        
        # unfold → (B*C, N, P)
        # Create overlapping/unoverlapping patches: (B*C, N, P)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B*C, N, P)
        Bc, N, P = patches.shape
        # Linear projection per patch
        tokens = self.proj(patches)  # (B*C, N, d_model)
        return tokens


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # x: (B, N, d_model)
        N = x.size(1)
        return x + self.pe[:, :N, :]

# ----------------------------------------
# Stage 1A: Sequence encoder (PatchTST CI)
# ----------------------------------------
class PatchTSTEncoder(nn.Module):
    """
    Channel-Independent Transformer over patches (shared weights across channels).
    - InstanceNorm per (sample, channel) series.
    - Patchify + linear embedding.
    - Positional encoding + TransformerEncoder.
    - Mean-pool tokens -> per-channel representation.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        patch_len: int,
        stride: int,
        use_batchnorm_out: bool = False
    ):
        super().__init__()
        self.inst_norm = nn.InstanceNorm1d(1, affine=False, eps=1e-6)

        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, d_model=d_model)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=n_heads,
        #     dim_feedforward=d_ff,
        #     dropout=dropout,
        #     activation="gelu",
        #     batch_first=True,  # (B, N, E)
        #     norm_first=True,
        # )

        encoder_layer = TransformerBatchNormEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_enc = SinusoidalPositionEncoding(d_model)

        # Use either batch normalization or layer normalization
        self.use_bn = use_batchnorm_out
        if self.use_bn:
            # BN over feature dim: expects (B, d_model, seq)
            self.bn_out = nn.BatchNorm1d(d_model)
        else:
            self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x):  # x: (B, C, L)
        B, C, L = x.shape

        # InstanceNorm per channel per sample
        x = x.reshape(B * C, 1, L)
        x = self.inst_norm(x)        # (B*C, 1, L)
        x = x.squeeze(1)             # (B*C, L)

        # Patching + embedding
        tokens = self.patch_embed(x) # (B*C, N, d_model)

        # Positional encoding + Transformer
        tokens = self.pos_enc(tokens)
        enc = self.encoder(tokens)   # (B*C, N, d_model)

        # # Mean pooling over tokens to get one vector per (sample, channel)
        # pooled = enc.mean(dim=1)     # (B*C, d_model)
        # pooled = self.norm_out(pooled)
        # # Reshape back to (B, C, d_model)
        # return pooled.view(B, C, -1)

        # reshape to group channels, then aggregate across channels
        BxC, N, D = enc.shape
        enc = enc.view(B, C, N, D)        # (B, C, N, d_model)
        enc = enc.mean(dim=1)             # (B, N, d_model)   <-- temporal tokens

        # optional norm
        if self.use_bn:
            enc = enc.transpose(1, 2)     # (B, d_model, N)
            enc = self.bn_out(enc)
            enc = enc.transpose(1, 2)     # (B, N, d_model)
        else:
            enc = self.ln_out(enc)        # (B, N, d_model)

        return enc  # temporal_out: (B, N, d_model)



# ----------------------------------------------
# Stage 1B: Feature encoder (channel attention)
# -----------------------------------------------
class SensorChannelTransformerEncoder(nn.Module):
    """
    Attend across sensors. For each sensor, compress its time window L -> d_model,
    yielding tokens = sensors (length C).
    """
    def __init__(self,  C: int, L: int, patch_len: int, stride: int, 
                 d_model=128, n_heads=8, num_layers=4, dim_feedforward=512, dropout=0.1,
                 use_batchnorm_out: bool = False):
        super().__init__()

        self.C = C
        self.L = L

        # Compress time dimension per sensor to d_model
        # self.embedding = nn.Linear(L, d_model)    # applied to last dim (time)

        # Patch embedding along the channel (sensor) dimension
        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, d_model=d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionEncoding(d_model)

        # # Transformer encoder layers
        # encoder_layers = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", batch_first=True
        # )

        encoder_layers = TransformerBatchNormEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # # Final norm
        # self.norm_out = nn.LayerNorm(d_model)

        # Norm at output
        # Use either batch normalization or layer normalization
        self.use_bn = use_batchnorm_out
        if self.use_bn:
            self.bn_out = nn.BatchNorm1d(d_model)  # will use (B, d_model, C)
        else:
            self.ln_out = nn.LayerNorm(d_model)

        # self.inst_norm = nn.InstanceNorm1d(self.C, affine=False, eps=1e-6)
        # (optional) IN across sensors for each time index; comment out if not wanted

    def forward(self, x):
        """
        x: (B, C, L)  -> sensor-time matrix
        We patch along the *sensor dimension C*.
        """
       
        B, C, L = x.shape
        assert C == self.C and L == self.L, "Shape mismatch for SensorChannelTransformerEncoder"

        # optional: instance-norm across sensors for each time index
        # expects (B, C, L); IN normalizes each channel across L, shared eps
        # x = self.inst_norm(x)       

        # compress time per sensor: (B, C, L) -> (B, C, d_model)
        # x = self.embedding(x)  # Linear over last dim

        # Rearrange to (B*L, C) so we can patch along channels
        x = x.permute(0, 2, 1)     # (B, L, C)
        x = x.reshape(B * L, C)    # treat each time step separately

        # Apply patch embedding along sensor dimension
        tokens = self.patch_embed(x)  # (B*L, num_patches, d_model)

        # Restore batch/time structure
        num_patches = tokens.size(1)
        tokens = tokens.view(B, L, num_patches, -1)   # (B, L, N_patch, d_model)

        # Merge time and sensor-patch tokens: treat each (time, patch) as a token
        tokens = tokens.view(B, L * num_patches, -1)  # (B, L*N_patch, d_model)

        # add sensor positional encodings: treat sensors as tokens
        tokens = self.pos_encoder(tokens)     # (B, C, d_model)

        # Transformer over sensor tokens
        enc = self.transformer_encoder(tokens)     # (B, L*N_patch, d_model)

        # final norm
        if self.use_bn:
            enc = enc.transpose(1, 2)   # (B, d_model, seq)
            enc = self.bn_out(enc)
            enc = enc.transpose(1, 2)   # (B, seq, d_model)
        else:
            enc = self.ln_out(enc)

        return enc   # (B, L*N_patch, d_model)


# -------------------------------------
# Fusion with Batch Normalization
# -------------------------------------
class FusionHead(nn.Module):
    def __init__(self, d_model_t: int, d_model_c: int, head_hidden: Optional[int] = None,
                 dropout: float = 0.1, pooling="mean"):
        super(FusionHead, self).__init__()

        # project to common width
        self.proj_t = nn.Identity() if d_model_t == d_model_c else nn.Linear(d_model_t, d_model_c)
        self.d_model = d_model_c

        # Replace LayerNorm with BatchNorm
        self.norm = nn.BatchNorm1d(self.d_model)

        assert pooling in ["mean", "cls"]
        self.pooling = pooling

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, temporal_out, channel_out):
        """
        temporal_out: (B, N, d_model_t)
        channel_out : (B, C, d_model_c)
        """
        
        # Project temporal side
        t = self.proj_t(temporal_out)  # (B, N, d_model)
        c = channel_out                # (B, C, d_model)

        # Concat token sequences
        p = torch.cat([t, c], dim=1)   # (B, N+C, d_model)

        # --- BatchNorm requires permute ---
        p = p.permute(0, 2, 1)         # (B, d_model, N+C)
        p = self.norm(p)               # BN across feature dimension
        p = p.permute(0, 2, 1)         # back to (B, N+C, d_model)

        # Pooling
        if self.pooling == "mean":
            pooled = p.mean(dim=1)     # (B, d_model)
        else:
            pooled = p[:, 0, :]        # CLS-style

        return self.mlp(pooled)
    


# Can use PatchTST_RUL_Model - as a single stage
class PatchTST_RUL_Model(nn.Module):
    def __init__(
        self,
        C, L, 
        d_model_t: int ,
        n_heads_t: int ,
        n_layers_t: int ,
        d_ff_t: int ,
        dropout_t: float ,
        patch_len_t: int ,
        stride_t: int ,
        patch_len_c: int ,
        stride_c: int ,
        d_model_c: int ,
        n_heads_c: int ,
        n_layers_c: int ,
        d_ff_c: int ,
        dropout_c: float,
        head_hidden: Optional[int] = None,
        pooling="mean",
        use_bn_temporal=True, 
        use_bn_channel=True
        
    ):
        super().__init__()

        self.temporal_encoder = PatchTSTEncoder(
            d_model=d_model_t, n_heads=n_heads_t, n_layers=n_layers_t,
            d_ff=d_ff_t, dropout=dropout_t, patch_len=patch_len_t, stride=stride_t, use_batchnorm_out=use_bn_temporal
        )

        self.sensor_encoder = SensorChannelTransformerEncoder(C=C, L=L, patch_len=patch_len_c, stride=stride_c,
            d_model=d_model_c, n_heads=n_heads_c, num_layers=n_layers_c, dim_feedforward=d_ff_c,
            dropout=dropout_c, use_batchnorm_out=use_bn_channel
        )

        self.fusion_head = FusionHead(d_model_t, d_model_c, head_hidden, dropout_t, pooling)

    def forward(self, x):  # x: (B, C, L)
        te = self.temporal_encoder(x)       # (B, N, d_model_t)
        se = self.sensor_encoder(x)         # (B, C, d_model_c)             
        y = self.fusion_head(te, se)        # (B, 1)

        return y.squeeze(-1)                # (B,)
    


# %%
def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R^2: 1 - SSE/SST (guard against zero variance edge case)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def train_one_epoch(model, loader, device, optimizer, criterion, grad_clip: float = 1.0):
    
    model.train()    
    total_loss = 0.0
    n = 0    
    counter = 1        
    num_batches = int(len(loader)) 

    for i, (xb, yb) in enumerate(loader):
        if i >= num_batches:   # stop after number of batches
            break
    # for xb, yb in loader:        
        if (counter < 3):
            print(f'Loading training data for batch ##:{counter}')

        xb = xb.to(device, non_blocking=True).float()  # (B, C, L)
        yb = yb.to(device, non_blocking=True).float()  # (B,)        
        optimizer.zero_grad(set_to_none=True)
        Bx, Cx, Lx = xb.shape
        preds = model(xb)   # (B,)
        loss = criterion(preds, yb)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        counter += 1
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses = 0.0
    n = 0
    preds_all = []
    targets_all = []

    num_batches = int(len(loader))


    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            if i >= num_batches:   # stop after number of batches
                break
        # for xb, yb in loader[:30]:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()
            preds = model(xb)
            loss = criterion(preds, yb)
            losses += loss.item() * xb.size(0)
            n += xb.size(0)
            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(yb.detach().cpu().numpy())
        y_pred = np.concatenate(preds_all, axis=0)
        y_true = np.concatenate(targets_all, axis=0)
        loss = losses / max(1, n)
        mets = metrics_regression(y_true, y_pred)
    return loss, mets, y_true, y_pred



@dataclass
class TrainConfig:
    feature_cols: List[str]
    target_col: str = "RUL"
    group_col: str = "engine"
    time_col: str = "time"

    # Sliding window
    C: int = len(features)
    L: int = window_size
    # lookback: int = window_size
    # step_size: int = 1
    patch_len_t: int = 5
    stride_t: int = 5
    patch_len_c: int = 5
    stride_c: int = 5

    # Model
    d_model_t: int = 64
    n_heads_t: int = 4
    n_layers_t: int = 4
    d_ff_t: int = 256
    dropout_t: float = 0.2
    head_hidden: Optional[int] = None
    use_feature_attn: bool = True
    d_model_c: int = 64
    n_heads_c: int = 4
    n_layers_c: int = 4
    d_ff_c: int = 256
    dropout_c: float = 0.2
    head_dropout: float = 0.1

    # Optim
    batch_size: int = 40
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10  # early stopping patience

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    model_path: str = "patchtst_rul_best.pt"

class RULWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, C, L) float32
        y: (N,)      float32
        """
        assert X.ndim == 3, "X must be (N, C, L)"
        assert y.ndim == 1, "y must be (N,)"
        # Ensure float32 (avoid float64 vs float32 mismatch)
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def make_loaders(X_train, X_val , y_train, y_val, batch_size = 64, num_workers = 0, use_cuda=torch.cuda.is_available()):    
   
    # Convert X into (N, C, L) and y into (N,)
    # X : np.ndarray of shape (N, C, L)    where C = len(feature_cols), L = lookback
    # y : np.ndarray of shape (N, )

    # Transpose last two dimensions
    X_train_trans = X_train.transpose(0, 2, 1)   
    X_val_trans = X_val.transpose(0, 2, 1)    

    train_ds = RULWindowDataset(X_train_trans , y_train)
    val_ds   = RULWindowDataset(X_val_trans , y_val)

    pin = bool(use_cuda)  # pin only if training on CUDA

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, (X_train_trans.shape[1], X_train_trans.shape[2])  # (C, L)


def make_test_loader(X_testf, y_test, batch_size = 64, num_workers = 0, use_cuda=torch.cuda.is_available()):
    # Convert X into (N, C, L) and y into (N,)
    # X : np.ndarray of shape (N, C, L)    where C = len(feature_cols), L = lookback
    # y : np.ndarray of shape (N, )

    # Transpose last two dimensions
    X_test_trans = X_testf.transpose(0, 2, 1)  

    test_ds  = RULWindowDataset(X_test_trans, y_test)
    pin = bool(use_cuda)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, len(test_ds)), shuffle=False, num_workers=num_workers, pin_memory=pin)
    return test_loader

# Data
# train_dataloader, val_dataloader, (C, L) = make_loaders(X_train, X_val , y_train, y_val, num_workers = 0, use_cuda=(TrainConfig.device.startswith("cuda")))
# test_dataloader = make_test_loader(X_testf, y_test, num_workers = 0, use_cuda=(TrainConfig.device.startswith("cuda")))

# print(f"Ready For Training Windowed shapes -> Channels: {C}, Lookback: {L}")



def fit_patchtst_dualdim_rul(train_dataloader, val_dataloader, test_dataloader,
                     features, cfg: TrainConfig):
    
    # Model - Single Stage
    model = PatchTST_RUL_Model(
        C=cfg.C,
        L=cfg.L,
        d_model_t=cfg.d_model_t,
        n_heads_t=cfg.n_heads_t,
        n_layers_t=cfg.n_layers_t,
        d_ff_t=cfg.d_ff_t,
        dropout_t=cfg.dropout_t,
        patch_len_t=cfg.patch_len_t,
        stride_t=cfg.stride_t,
        patch_len_c=cfg.patch_len_c,
        stride_c=cfg.stride_c,
        d_model_c=cfg.d_model_c,
        n_heads_c=cfg.n_heads_c,
        n_layers_c=cfg.n_layers_c,
        d_ff_c=cfg.d_ff_c,
        dropout_c=cfg.dropout_c,
        head_hidden=cfg.head_hidden,
        pooling="mean",
        use_bn_temporal=True,
        use_bn_channel=True
    ).to(device)
    

    # Optimizer/loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    epochs_no_improve = 0

    # Collect the train and test loss per epoch
    train_losses = []
    val_losses = []

    epoch_num=1

    for epoch in range(1, cfg.epochs + 1):
        print(f'Epoch #:{epoch_num} out of {cfg.epochs}')
        t0 = time.time()       
        train_loss = train_one_epoch(model, train_dataloader, device, optimizer, criterion, cfg.grad_clip)
        print('evaluating model')
        val_loss, val_mets, _, _ = evaluate(model, val_dataloader, device, criterion)
        epoch_time = time.time() - t0

        # Step the scheduler
        scheduler.step(val_loss)

        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch:03d}] "
              f"TrainLoss={train_loss:.4f}  "
              f"ValLoss={val_loss:.4f}  "
              f"ValMAE={val_mets['MAE']:.4f}  ValRMSE={val_mets['RMSE']:.4f}  ValR2={val_mets['R2']:.4f}  "
              f"({epoch_time:.1f}s)")

        # Early stopping on MAE
        if val_mets['MAE'] < best_val_mae - 1e-6:
            best_val_mae = val_mets['MAE']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch}. Best Val MAE: {best_val_mae:.4f}")
                break

        epoch_num +=1

    # Save best
    if best_state is not None:
        torch.save(best_state, cfg.model_path)
        print(f"Saved best model to: {cfg.model_path}")

    # Load best for final eval
    if best_state is not None:
        model.load_state_dict(best_state)

    
    return model, train_losses, val_losses


# %%
# Function to compute score 
def score(errors):
  a1=10
  a2=13
  s1=0
  s2=0
  for err in errors:
    if err < 0:
      s1 += (np.exp(-1*(err/a1))) - 1
    if ((err > 0) or (err == 0)):
      s2 += (np.exp(err/a2)) - 1
  return [s1 , s2]

# Data
# Training data set
import optuna
import random
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
best_rmse = 20

# Generate a list of 20 random integers between 0 and 200
rand_cnt = 1
random_state_lst = [42]
# random_state_lst = [random.randint(0, 500) for _ in range(30)]


def objective(trial):
    global X_train_sw, y_train_sw, X_train, X_val, y_train, y_val, X_testf, y_test

    # ---- Define hyperparameter search space ----
    cfg = TrainConfig(
        feature_cols = features,
        target_col = "rul",
        group_col = "engine",
        time_col = "time",

        # Sliding window
        C = len(features),
        L = window_size,     
        patch_len_t = trial.suggest_int("patch_len_t", 5, 40, step=5),
        stride_t = trial.suggest_int("stride_t", 4, 12, step=2),
        patch_len_c = len(features),
        stride_c = len(features)//2,

        # Model hyperparams
        d_model_t = trial.suggest_categorical("d_model_t", [32, 64, 128]),
        n_heads_t = trial.suggest_categorical("n_heads_t", [2, 4, 8]),
        n_layers_t = trial.suggest_int("n_layers_t", 1, 4),
        d_ff_t = trial.suggest_categorical("d_ff_t", [64, 128, 256]),
        dropout_t = trial.suggest_float("dropout_t", 0.05, 0.3),

        d_model_c = trial.suggest_categorical("d_model_c", [32, 64, 128]),
        n_heads_c = trial.suggest_categorical("n_heads_c", [2, 4, 8]),
        n_layers_c = trial.suggest_int("n_layers_c", 1, 4),
        d_ff_c = trial.suggest_categorical("d_ff_c", [64, 128, 256]),
        dropout_c = trial.suggest_float("dropout_c", 0.05, 0.3),

        head_hidden = trial.suggest_categorical("head_hidden", [64, 128, 256]),
        head_dropout = trial.suggest_float("head_dropout", 0.05, 0.3),

        # Optimization parameters
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]),
        epochs = 150,  # fewer epochs for tuning speed
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        grad_clip = 1.0,
        patience = 10,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        num_workers = 0,
        model_path = f"optuna_best_trial_{trial.number}_{eng_type}.pt",
    )  

    # random_state_lst = [random.randint(0, 500) for _ in range(3)]
    # for randstate in random_state_lst:    
    randstate = 101
    X_train, X_val , y_train, y_val  = train_test_split(X_train_sw, y_train_sw, test_size=0.2 , random_state=randstate)

    train_dataloader, val_dataloader, (C, L) = make_loaders(X_train, X_val , y_train, y_val, num_workers = 0, use_cuda=(TrainConfig.device.startswith("cuda")))
    test_dataloader = make_test_loader(X_testf, y_test, num_workers = 0, use_cuda=(TrainConfig.device.startswith("cuda")))

    print(f"Ready For Training Windowed shapes -> Channels: {C}, Lookback: {L}; Attempt {rand_cnt} out of {len(random_state_lst)}")

    # make device variable for functions that expect it
    device = torch.device(cfg.device)
    # put device into global namespace expected by fit/evaluate if necessary
    globals()['device'] = device

    model, train_losses, val_losses = fit_patchtst_dualdim_rul(train_dataloader, val_dataloader, test_dataloader,features, cfg)

    # Test evaluation
    criterion = nn.MSELoss()
    test_loss, test_mets, y_test, y_predb = evaluate(model, test_dataloader, device, criterion)
    print(f"[TEST]  Loss={test_loss:.4f}  "
            f"MAE={test_mets['MAE']:.4f}  RMSE={test_mets['RMSE']:.4f}  R2={test_mets['R2']:.4f}")
    test_report = {"loss": test_loss, **test_mets}

        # Keep metrics as floats (don't prematurely round)
    mae = float(test_mets['MAE'])
    rmse = float(test_mets['RMSE'])
    r2 = float(test_mets['R2'])

    # Score metrics
    errors = y_predb - y_test
    scores = score(errors)
    print(f"Score Metric: S1 = {scores[0]:.2f} S2 = {scores[1]:.2f}")

    # Report result to Optuna
    trial.report(rmse, step=0)

    # Prune unpromising trials early
    if trial.should_prune():
        raise optuna.TrialPruned()

    # if (test_mets['RMSE'] < best_rmse):            
    #         best_model = model
    #         best_scores = scores
    #         best_r_sq = test_mets['R2']
    #         best_mae = test_mets['MAE']
    #         best_rmse = test_mets['RMSE']
    #         best_train_losses = train_losses
    #         best_val_losses = val_losses
    #         best_randon_state = randstate
    
    # rand_cnt += 1
    return rmse

study_name = f"patchtst_dualdim_rul_optuna_{eng_type}"
storage_name = f"sqlite:///{study_name}.db"  # saves trials persistently

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction="minimize",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
)

study.optimize(objective, n_trials=25, timeout=None)  # run for 3 hours max  , timeout=60*60*3

print("Best trial:")
best_trial = study.best_trial
print(f"  Value (RMSE): {best_trial.value:.4f}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Save study with a consistent filename (use eng_type if you want)
study_pkl = f"optuna_patchtst_dualdim_rul_{eng_type}.pkl"
joblib.dump(study, study_pkl)


# print('************************')
# best_model_path = f'dualdimen_patchtst_rul_best_rand_selected_{eng_type}.pt'
# best_state = copy.deepcopy(best_model.state_dict())
# torch.save(best_state, best_model_path)
# print(f"Saved best model to: {best_model_path}")
# print(f'Best Scores: S1 = {best_scores[0]:.2f} S2 = {best_scores[1]:.2f}')
# print(f'Best random state was: {best_randon_state}')
# print(f'Best RMSE: {best_rmse:.4f}')
# print(f'Best MAE: {best_mae:.4f}')
# print(f'Best R2: {best_r_sq:.4f}')

# %%
# --- After tuning: load study and build cfg from best params ---
loaded_study = joblib.load(study_pkl)
best_params = loaded_study.best_params

base_cfg = TrainConfig(
    feature_cols = features,
    target_col = "rul",
    group_col = "engine",
    time_col = "time",

    # Sliding window
    C = len(features),
    L = window_size,     
    patch_len_t = 30,
    stride_t = 8,
    patch_len_c = len(features),
    stride_c = len(features)//2,

    # Model
    d_model_t = 64,
    n_heads_t = 4,
    n_layers_t = 2,
    d_ff_t = 128,
    dropout_t = 0.1,
    use_feature_attn = True,
    d_model_c = 64,
    n_heads_c = 4,
    n_layers_c = 2,
    d_ff_c = 128,
    dropout_c = 0.1,
    head_hidden = 128,
    head_dropout = 0.1,

    # Optim
    batch_size = 64,
    epochs = 150,
    lr = 1e-4,
    weight_decay = 1e-4,
    grad_clip = 1.0,
    patience = 15,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers = 0,
    model_path = f"dualdimen_patchtst_rul_best_{eng_type}.pt",
)

# Merge defaults + best params
cfg = TrainConfig(**{**vars(base_cfg), **best_params})

random_state=333
X_train, X_val , y_train, y_val  = train_test_split(X_train_sw, y_train_sw, test_size=0.2 , random_state=random_state)

train_dataloader, val_dataloader, (C, L) = make_loaders(X_train, X_val , y_train, y_val, num_workers = 0, use_cuda=(TrainConfig.device.startswith("cuda")))
test_dataloader = make_test_loader(X_testf, y_test, num_workers = 0, use_cuda=(TrainConfig.device.startswith("cuda")))


best_model, train_losses, val_losses = fit_patchtst_dualdim_rul(
    train_dataloader, val_dataloader, test_dataloader, features, cfg
)


# %%
# Evaluate the best model
criterion = nn.MSELoss()
test_loss, test_mets, y_test, y_predb = evaluate(best_model, test_dataloader, device, criterion)



# %%
# Helper function that plots, saves and shows plots
def plot_loss_accuracy(filename, train_losses, val_losses):
  loss = train_losses
  val_loss = val_losses

  epochs = range(len(loss))

  plt.plot(epochs, loss, 'r', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')

  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.grid(True)
  plt.legend(loc=0)
  plt.title('Training and validation loss')
  plt.tight_layout()
  plt.figure()

  plt.savefig(filename + '.png')
  plt.show()

# Plot Loss
plot_loss_accuracy("Transformer", train_losses=train_losses, val_losses=val_losses)


# %%
# y_predb = scaler.inverse_transform(y_predb.reshape(-1,1))
# y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# %%
# Optional: Plot Actual vs. Predicted RUL
plt.figure(figsize=(10, 5))
plt.plot(y_test.flatten(), label="Actual RUL", linestyle="dashed", color="blue")
plt.plot(y_predb.flatten(), label="Predicted RUL", color="red")
plt.xlabel("Test Sample Index")
plt.ylabel("Remaining Useful Life (RUL)")
plt.legend()
plt.title("Actual vs Predicted RUL")
plt.show()


# Plot Predicted vs Actual RUL
sns.scatterplot(x = y_test.flatten() , y = y_predb.flatten()  , s = 30 , alpha = 0.5)

sns.lineplot( x = [ min(y_test.flatten())  , max(y_test.flatten()) ] , 
             y = [min(y_test.flatten()) , max(y_test.flatten())] , color = 'red')
 

plt.xlabel('TRANSFORMER Predicted')
plt.ylabel('Actual')

# Calculate performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test.flatten(), y_predb.flatten())
mse = mean_squared_error(y_test.flatten(), y_predb.flatten())
rmse = np.sqrt(mse)
r2 = r2_score(y_test.flatten(), y_predb.flatten())



# %%
ind = np.argsort(-y_test)
fig = plt.figure(figsize=(10, 6))
plt.plot(y_test[ind])
plt.plot(y_predb[ind], 'ro-')
# plt.plot(y_predb[ind])
plt.title('Actual vs Predicted RUL')
plt.xlabel('Index')
plt.ylabel('Remaining Useful Life')
plt.show()

# %%
# Print the results
errors = y_predb - y_test
best_scores = score(errors)
# f"MAE={test_mets['MAE']:.4f}  RMSE={test_mets['RMSE']:.4f}  R2={test_mets['R2']:.4f}")
best_total_score = best_scores[0] + best_scores[1]
print(f'Model Performamce Results on Engine Type: ',eng_type)
print("Root Mean Squared Error (RMSE): ", round(test_mets['RMSE'],2))
print(f"Score Metric: S1 = {best_scores[0]:.2f} S2 = {best_scores[1]:.2f}")
print(f"Total Score: {best_total_score:.2f}")
print("Mean Absolute Error (MAE): ", round(test_mets['MAE'],2))
print("Mean Squared Error (MSE): ", round(mse,2))
print("R2 Score: ", round(test_mets['R2'],4))
print(f'Random State: {random_state_lst[0]}')


# # Print the results

# # f"MAE={test_mets['MAE']:.4f}  RMSE={test_mets['RMSE']:.4f}  R2={test_mets['R2']:.4f}")
# print(f'Model Performamce Results on Engine Type: ',eng_type)
# print(f'Best random state was: {best_randon_state}')
# print("Root Mean Squared Error (RMSE): ", round(test_mets['RMSE'],2))
# print(f"Score Metric: S1 = {scores[0]:.2f} S2 = {scores[1]:.2f}")
# print("Mean Absolute Error (MAE): ", round(test_mets['MAE'],2))
# # print("Mean Squared Error (MSE): ", round(mse,2))
# print("R2 Score: ", round(test_mets['R2'],4))

# %%
print(f"Best Random State: {random_state}")
# %%
