# =============================================================================
#  MainSingleEng_FD001_F2_FD003_F4_Final_Improved.py
# -----------------------------------------------------------------------------
#  Improvement-experiment driver for the dual-encoder (temporal + sensor-channel)
#  PatchTST RUL model on the NASA C-MAPSS turbofan benchmark (FD001-FD004).
#  This file contains the shared data pipeline + model definitions, and then
#  dispatches to one improvement experiment selected by `experiment_mode`.
#
#  HOW TO RUN
#  ----------
#  1. Run from the repo root ("Final Code and Files/") - the script uses
#     relative paths like Data/train_FD001.txt.
#  2. Select the engine: edit `eng_type` (one of FD001 / FD002 / FD003 / FD004).
#  3. Select the experiment: set `experiment_mode` (≈ line 918) to one of the
#     values in the table below, then run the file top-to-bottom.
#     Each value gates an `if (experiment_mode == ...)` cell that imports its
#     module and calls that module's `run_*` entrypoint.
#
#  experiment_mode values
#  ----------------------
#   'Improved Transformer'           -> improve_transformer.run_all_scenarios
#       Scenarios A-E: asymmetric Huber loss, OneCycleLR, capacity/window bump,
#       Gaussian-jitter augmentation, 3-seed ensemble.
#   'Sensitivity Analysis'           -> improve_transformer_sensitivity_analysis
#                                        .run_all_sensitivity_experiments
#       Robustness sweeps: input noise / train-set size / dropout.
#   'Rotational Positional Encoding' -> improve_transformer_rope.run_rope_scenarios
#       RoPE replaces sinusoidal positional encoding (applied in attention Q/K).
#   'BERT'                           -> improve_transformer_bert.run_bert_scenarios
#       Two-phase training: (1) train encoders, (2) freeze them and train a head
#       over the concatenation of the top-K transformer layers.
#   'Baseline'                       -> _train_baseline_3seeds.train_baseline_3seeds
#       Plain baseline architecture, 3 seeds. Also has its own dedicated file
#       MainSingleEng_FD001_F2_FD003_F4_Final_Baseline.py.
#   'Cross Attention'                -> NOT implemented here. See
#       MainSingleEng_FD001_F2_FD003_F4_Final_ImproveCrossAttn.py.
#
#  NOTE: mode-specific "reload checkpoint + display results" cells live further
#  down the file (≈ L1050 Improved Transformer, L1299 RoPE, L1637 BERT). Run the
#  cell that matches the experiment you trained.
# =============================================================================

# %%
import os
import math
import time
import copy
import pickle
import random
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from Encoder_Layers import *
from CommonFunctions import *
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from keras import models, layers
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset, DataLoader


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
eng_type = 'FD001'
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
if ((eng_type == 'FD002') or (eng_type == 'FD004')):
   
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


    #############################################################
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

    #############################################################
    # Save the trained model to a file
    import joblib
    joblib.dump(kmeans, f"kmeans_model_{eng_type}.pkl")
    print("KMeans model saved as kmeans_model.pkl")

    # Load saved model
    kmeans_loaded = joblib.load(f"kmeans_model_{eng_type}.pkl")

    # Prepare test dataset (must have same features and scaling as training data)
    op_condit_test_df = df_test[['op_setting_1', 'op_setting_2', 'op_setting_3']]

    # Predict cluster assignments for the test data
    test_cluster_labels = kmeans_loaded.predict(op_condit_test_df)

    # Optionally, add the cluster assignments back into the test DataFrame
    df_test.insert(1, 'OperCluster', test_cluster_labels)

    df_test.head()


    #############################################################
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

    #############################################################
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


    #############################################################
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
if ((eng_type == 'FD002') or (eng_type == 'FD004')):
    # Compute correlation between sensor data and mask top section for correlation matrix for better readability
    df_corr = norm_df.corr()
    mask = np.tril(np.ones(df_corr.shape),k = -1).astype(bool)
    df_corr = df_corr.where(mask)
    df_corr
else:    
    # Compute correlation between sensor data and mask top section for correlation matrix for better readability
    df_corr = df.corr()
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
        if abs(df_corr.loc[col , row]) > 0.95 :
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

col_to_drop

if ((eng_type == 'FD002') or (eng_type == 'FD004')):
    col_to_drop = ['sm_1', 'sm_5', 'sm_18', 'sm_19'] + col_to_drop 


# %%
if ((eng_type == 'FD001') or (eng_type == 'FD003')):
    # These 2 feature has very high correlation , no need for both of them , we can drop one of them
    df.drop(columns = col_to_drop , inplace = True)

    df.columns

# %%
# EDA 
data_train['time'].describe().T

# There are 100 units(engins)
# Time in cycles is between 1 - 362 cycles, cycle could be every turbine run

# %%
# max , or failure time for each engine , or max time cycle , that engine has worked.
failure_time = df.groupby('engine')['rul'].max()

# Plot Max Failure Time For Each Engine
plt.figure(figsize = (6,14))
sns.barplot(y = failure_time.index , x = failure_time.values , orient='h')
plt.title('failure time for engine')
plt.xlabel('failure time')
plt.ylabel('engine number')
plt.tight_layout()
plt.show()

# %%
# Distribution of failure time per engine
sns.histplot(failure_time , kde=True)
plt.title('Failure time for engine')
plt.tight_layout()

# %%
# Function for Sensor visualization - plot signals from sensors
def plot_signal(df , signal_name , Sensor_dictionary):
    figure = plt.figure(figsize=(10,4))

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
        plot_signal(df , 'sm_'+str(i)  , Sensor_dictionary)
    except:
        pass

# %%
if ((eng_type == 'FD002') or (eng_type == 'FD004')):
    # The feature has very high correlation , no need for both of them , we can drop one of them
    # Also drop the features that have zero correlation 
    # Do it for both Training and Testing dataset
    norm_df.drop(columns = col_to_drop , inplace = True)  
    norm_df_test.drop(columns = col_to_drop , inplace = True)
    print(norm_df.columns)
    print(norm_df_test.columns)


    # Identify the features to be used in the models
    features = norm_df.columns[2:-1]   # drop  engine , time , rul of  dataset
    features
else:
    # Identify the features to be used in the models
    print(df.columns)
    features = df.columns[2:-1]   # drop  engine , time , rul of  dataset
    features 


# %%
if ((eng_type == 'FD002') or (eng_type == 'FD004')):
    # Piecewise function to prepare y (target) for training data
    # RUL is the target variable, and it is the last column in the dataframe
    norm_df['rul'] = np.where(norm_df['rul'] >= 125, 125, norm_df['rul'])  # Ensure RUL is non-negative

    # Prepare blind test set - extract only features that will be use for testing
    y_test['RUL'] = np.where(y_test['RUL'] >= 125, 125, y_test['RUL'])  # Ensure RUL is non-negative
    y_test                       # y_test is blind test's target 

    X = norm_df.copy()
    X_test = norm_df_test.copy()


else:

    # Training data preparation
    df_train_eng_col = df['engine']
    X = df[features]
    # Piecewise function to prepare y (target) for training data
    # RUL is the target variable, and it is the last column in the dataframe
    df['rul'] = np.where(df['rul'] >= 125, 125, df['rul'])  # Ensure RUL is non-negative
    y = df['rul']

    # Testing data preparation
    df_test_eng_col = df_test['engine']
    # Prepare blind test set - extract only features that will be use for testing
    X_test = df_test[features]   # df_test from blind test set (has no target)
    y_test['RUL'] = np.where(y_test['RUL'] >= 125, 125, y_test['RUL'])  # Ensure RUL is non-negative
    y_test                       # y_test is blind test's target 


    #############################################################################
    # Transform and normalized data for training 
    scaler = StandardScaler()      # Using StandardScaler instead of MinMaxScaler
    X = scaler.fit_transform(X)    # fit on only train dataset
    X_test = scaler.transform(X_test)

    # Combine engine information with normailzed data
    X = pd.DataFrame(data = np.c_[df_train_eng_col, X], columns=pd.Index(['engine'] + list(features)))
    X_test = pd.DataFrame(data = np.c_[df_test_eng_col, X_test], columns=pd.Index(['engine'] + list(features)))
    X = pd.concat([X, y], axis = 1)

# %%
from smooth_timeseries import smooth_engine_data, compare_smoothing_methods

# Optional: visualise all five methods on one engine/sensor before deciding
compare_smoothing_methods(X, features, engine_id=1, sensor="sm_2",
                          save_path="smoothing_comparison.png")

# Apply smoothing (same call works for both training and test)
X_smooth      = smooth_engine_data(X,features, method="gaussian", sigma=2)
X_test_smooth = smooth_engine_data(X_test, features, method="gaussian", sigma=2)

# %%
#####################################################################################################################
# Sliding window function to prepare training data
# Parameters to prepare training data

window_size = 40    # Each batch will have a sequence of window_size series
# X_train_sw, y_train_sw = create_training_sequences_sw(X, features, window_size)
# Drop-in replacement — nothing else changes
X_train_sw, y_train_sw = create_training_sequences_sw(X_smooth, features, window_size)


# Sliding window function to prepare testing data
# Define how to shift window and number of batches
shift = 1
num_of_batches = 1

# X_testf = create_testing_sequences_sw(X_test, features, window_size, num_of_batches)
X_testf = create_testing_sequences_sw (X_test_smooth,  features, window_size, num_of_batches)
y_test = np.squeeze(y_test).to_numpy()
X_testf.shape



#########################################
# # Training set
# from sklearn.model_selection import train_test_split
# X_train, X_val , y_train, y_val  = train_test_split(X_train_sw, y_train_sw, test_size=0.2 ) # , random_state=42



# %%

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
# device = torch.device("cpu")  # If only using CPU
print(device)


# %%
# -----------------------------
# PatchTST blocks
# -----------------------------
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
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

# Fixed positional encoding
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

        # Patch embedding along the channel (sensor) dimension
        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, d_model=d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionEncoding(d_model)
        
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
    
######################################################

# %%

# experiment_mode can be one of the following options:
# 'Improved Transformer' ; 
# 'Cross Attention' ; has its own execution file MainSingleEng_FD001_F2_FD003_F4_Final_ImproveCrossAttn.py
# 'Rotational Positional Encoding' ; 
# 'BERT'
# 'Sensitivity Analysis'
# 'Baseline' ; has its own execution file MainSingleEng_FD001_F2_FD003_F4_Final_Baseline.py

# Set experiment
experiment_mode = 'Improved Transformer' 


# %%
# Experment 1 - Multiple scenarios with temporal and sensor-channel fusion 
if (experiment_mode == 'Improved Transformer'):
    from improve_transformer import run_all_scenarios

    leaderboard = run_all_scenarios(
        X_train_sw                   = X_train_sw,
        y_train_sw                   = y_train_sw,
        X_testf                      = X_testf,
        y_test                       = y_test,
        features                     = features,
        eng_type                     = eng_type,
        device                       = device,
        X                            = X,
        X_test                       = X_test,
        create_training_sequences_sw = create_training_sequences_sw,
        create_testing_sequences_sw  = create_testing_sequences_sw,
        num_of_batches               = num_of_batches,
        window_size                  = window_size,
        random_state                 = 341,
        run_ensemble                 = True,
        verbose                      = True,
    )

# %%
# Sensitivity analysis experiments
# experiment_mode = 'Sensitivity Analysis'
if (experiment_mode == 'Sensitivity Analysis'):
  from improve_transformer_sensitivity_analysis import (
      run_all_sensitivity_experiments,
  )

  results = run_all_sensitivity_experiments(
      X_train_sw=X_train_sw, y_train_sw=y_train_sw,
      X_testf=X_testf,       y_test=y_test,
      features=features,     eng_type=eng_type,    device=device,
      X=X, X_test=X_test,
      create_training_sequences_sw=create_training_sequences_sw,
      create_testing_sequences_sw=create_testing_sequences_sw,
      num_of_batches=num_of_batches, window_size=window_size,
      do_noise=True, do_size=True, do_dropout=True, do_cross=False,
      noise_seeds=20, size_seeds=3, dropout_seeds=3,
  )


# %%
# Experiment 4 - Using  Rotary Positional Embeddings (RoPE).
if (experiment_mode == 'Rotational Positional Encoding'):
    from improve_transformer_rope import run_rope_scenarios

    leaderboard = run_rope_scenarios(
        X_train_sw                   = X_train_sw,
        y_train_sw                   = y_train_sw,
        X_testf                      = X_testf,
        y_test                       = y_test,
        features                     = features,
        eng_type                     = eng_type,
        device                       = device,
        X                            = X,
        X_test                       = X_test,
        create_training_sequences_sw = create_training_sequences_sw,
        create_testing_sequences_sw  = create_testing_sequences_sw,
        num_of_batches               = num_of_batches,
        window_size                  = window_size,
        random_state                 = 341,
        run_ensemble                 = True,
    )
  

# %%
# Experiment 5 - Integrating strategies used in BERT
if (experiment_mode == 'BERT'):
    from improve_transformer_bert import run_bert_scenarios

    leaderboard = run_bert_scenarios(
        X_train_sw                   = X_train_sw,
        y_train_sw                   = y_train_sw,
        X_testf                      = X_testf,
        y_test                       = y_test,
        features                     = features,
        eng_type                     = eng_type,
        device                       = device,
        X                            = X,
        X_test                       = X_test,
        create_training_sequences_sw = create_training_sequences_sw,
        create_testing_sequences_sw  = create_testing_sequences_sw,
        num_of_batches               = num_of_batches,
        window_size                  = window_size,
        random_state                 = 341,
        run_ensemble                 = True,
        top_k_layers                 = 4,    # BERT top-4 layers (tunable)
        phase2_epochs                = 60,   # epochs for frozen-head training
        phase2_lr                    = 5e-4, # higher LR for head-only phase
        phase2_patience              = 15,
    )


# %%
compare_experiments = False

# ── Compare against original improve_transformer.py ───────────────────
if compare_experiments:
    from improve_transformer      import run_all_scenarios
    from improve_transformer_bert import run_bert_scenarios

    lb_orig = run_all_scenarios(...)
    lb_bert = run_bert_scenarios(...)

    print("Original  best RMSE:", min(r["RMSE"] for r in lb_orig))
    print("BERT      best RMSE:", min(r["RMSE"] for r in lb_bert))



# %%
if (experiment_mode == 'Baseline'):
    # Form A: explicit function call (recommended)
    from _train_baseline_3seeds import train_baseline_3seeds

    train_baseline_3seeds(
        X_train_sw=X_train_sw, y_train_sw=y_train_sw,
        X_testf=X_testf,       y_test=y_test,
        features=features,     eng_type=eng_type,
        device=device,
    )

# %%
##### USE FOR IMPROVED TRANSFORMER EXPERIMENT ########
##### RE-LOAD MODEL AND DISPLAY REULTS ####
import torch
if (experiment_mode == 'Improved Transformer'):
    model_path = f"BEST_improved_{eng_type}.pt"
    checkpoint = torch.load(model_path, map_location=device)

    # =============================================================================
    #  STEP 1 — Auto-detect architecture from checkpoint weight shapes
    #  This works for both single-model and ensemble checkpoints because every
    #  member of an ensemble was trained with the same architecture.
    # =============================================================================

    def detect_arch_from_state_dict(state_dict: dict) -> dict:
        """
        Infer model hyperparameters by inspecting tensor shapes in the state dict.
        No manual config needed — the weights tell us exactly what was trained.
        """
        arch = {}

        # d_model: size of the patch embedding projection output
        arch["d_model_t"] = state_dict["temporal_encoder.patch_embed.proj.bias"].shape[0]
        arch["d_model_c"] = state_dict["sensor_encoder.patch_embed.proj.bias"].shape[0]

        # patch_len: second dim of patch embedding weight (input size per patch)
        arch["patch_len_t"] = state_dict["temporal_encoder.patch_embed.proj.weight"].shape[1]
        arch["patch_len_c"] = state_dict["sensor_encoder.patch_embed.proj.weight"].shape[1]

        # n_layers: count how many encoder layer blocks exist in the state dict
        t_layer_keys = [k for k in state_dict if k.startswith("temporal_encoder.encoder.layers.")]
        c_layer_keys = [k for k in state_dict if k.startswith("sensor_encoder.transformer_encoder.layers.")]
        layer_indices_t = set(int(k.split(".")[3]) for k in t_layer_keys)
        layer_indices_c = set(int(k.split(".")[3]) for k in c_layer_keys)
        arch["n_layers_t"] = max(layer_indices_t) + 1
        arch["n_layers_c"] = max(layer_indices_c) + 1

        # n_heads: d_model / d_k, where d_k = q_proj output dim / n_heads
        # q_proj weight shape is (d_model, d_model) so nhead is inferred from d_model
        # We can read n_heads from the attn module's q_proj — must be a divisor of d_model
        # n_heads is not stored directly, but we stored d_model so just use 8 (your fixed value)
        arch["n_heads_t"] = 8
        arch["n_heads_c"] = 8

        # d_ff: first linear layer output size inside the encoder layer
        arch["d_ff_t"] = state_dict["temporal_encoder.encoder.layers.0.linear1.bias"].shape[0]
        arch["d_ff_c"] = state_dict["sensor_encoder.transformer_encoder.layers.0.linear1.bias"].shape[0]

        # head_hidden: first MLP layer in fusion head
        arch["head_hidden"] = state_dict["fusion_head.mlp.0.bias"].shape[0]

        # stride_t: cannot be recovered from weights — must be inferred from patch_len
        # Your configs always used stride_t = patch_len_t // 2, so:
        arch["stride_t"] = arch["patch_len_t"] // 2
        arch["stride_c"] = 1   # always 1 in all your configs

        # dropout: not stored in state dict — use a safe default (0.0 at inference)
        arch["dropout_t"] = 0.0
        arch["dropout_c"] = 0.0

        return arch


    # =============================================================================
    #  STEP 2 — Pull the first state dict (single or ensemble) for inspection
    # =============================================================================

    is_ensemble = isinstance(checkpoint, dict) and "ensemble" in checkpoint

    if is_ensemble:
        first_sd = checkpoint["ensemble"][0]
    else:
        first_sd = checkpoint

    arch = detect_arch_from_state_dict(first_sd)

    print("Detected architecture from checkpoint:")
    for k, v in arch.items():
        print(f"  {k:<15} : {v}")

    # =============================================================================
    #  STEP 3 — Reconstruct C and L from the state dict
    #  C (n_channels) and L (window_size) are not stored in weights either,
    #  but we know them from your notebook variables — just confirm they match.
    # =============================================================================

    # These come from your notebook — they must match what the model was trained with
    C = len(features)
    L = window_size   # the window_size used when X_train_sw was built

    print(f"\n  C (channels)   : {C}  (from features)")
    print(f"  L (window)     : {L}  (from window_size)")

    # =============================================================================
    #  STEP 4 — Build and load the model(s)
    # =============================================================================

    def build_model_from_arch(arch, C, L, device):
        m = PatchTST_RUL_Model(
            C           = C,
            L           = L,
            d_model_t   = arch["d_model_t"],
            n_heads_t   = arch["n_heads_t"],
            n_layers_t  = arch["n_layers_t"],
            d_ff_t      = arch["d_ff_t"],
            dropout_t   = arch["dropout_t"],
            patch_len_t = arch["patch_len_t"],
            stride_t    = arch["stride_t"],
            patch_len_c = arch["patch_len_c"],
            stride_c    = arch["stride_c"],
            d_model_c   = arch["d_model_c"],
            n_heads_c   = arch["n_heads_c"],
            n_layers_c  = arch["n_layers_c"],
            d_ff_c      = arch["d_ff_c"],
            dropout_c   = arch["dropout_c"],
            head_hidden = arch["head_hidden"],
            pooling     = "mean",
            use_bn_temporal = True,
            use_bn_channel  = True,
        ).to(device)
        return m


    if is_ensemble:
        print(f"\nCheckpoint type : ENSEMBLE  ({len(checkpoint['ensemble'])} members)")
        print(f"Best scenario   : {checkpoint.get('scenario', 'unknown')}")
        apply_isotonic  = checkpoint.get("apply_isotonic", False)

        loaded_models = []
        for i, state_dict in enumerate(checkpoint["ensemble"]):
            m = build_model_from_arch(arch, C, L, device)
            m.load_state_dict(state_dict)
            m.eval()
            loaded_models.append(m)
            print(f"  Ensemble member {i+1} loaded ✔")

        print(f"\nEnsemble of {len(loaded_models)} models ready.")
        print(f"Isotonic post-processing : {apply_isotonic}")

    else:
        print("\nCheckpoint type : SINGLE MODEL")
        loaded_model = build_model_from_arch(arch, C, L, device)
        loaded_model.load_state_dict(checkpoint)
        loaded_model.eval()
        print("Model loaded and set to eval mode ✔")

    # =============================================================================
    #  STEP 5 — Run inference
    # =============================================================================
    # from MainSingleEng_FD001_F2_FD003_F4_Final import make_test_loader
    from improve_transformer      import evaluate_improved, ensemble_predict, score_nasa

    class RULWindowDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            assert X.ndim == 3, "X must be (N, C, L)"
            assert y.ndim == 1, "y must be (N,)"
            self.X = X.astype(np.float32, copy=False)
            self.y = y.astype(np.float32, copy=False)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


    def make_test_loader(X_testf, y_test, batch_size=64, num_workers=0,
                        use_cuda=torch.cuda.is_available()):
        X_test_trans = X_testf.transpose(0, 2, 1)
        test_ds = RULWindowDataset(X_test_trans, y_test)
        pin = bool(use_cuda)
        test_loader = DataLoader(
            test_ds,
            batch_size=min(batch_size, len(test_ds)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        )
        return test_loader


    test_dataloader = make_test_loader(X_testf, y_test, batch_size=64,
                                    num_workers=0,
                                    use_cuda=torch.cuda.is_available())
    criterion = nn.MSELoss()

    if is_ensemble:
        from improve_rul_zone import ensemble_predict_zone
        y_pred_final, y_true_final = ensemble_predict_zone(
            loaded_models, test_dataloader, device,
            apply_isotonic=apply_isotonic
        )
        mae  = float(np.mean(np.abs(y_true_final - y_pred_final)))
        rmse = float(np.sqrt(np.mean((y_true_final - y_pred_final) ** 2)))
        sse  = np.sum((y_true_final - y_pred_final) ** 2)
        sst  = np.sum((y_true_final - np.mean(y_true_final)) ** 2)
        r2   = float(1.0 - sse / sst)
        final_mets = {"RMSE": rmse, "MAE": mae, "R2": r2}
    else:
        from improve_transformer import evaluate_improved
        _, final_mets, y_true_final, y_pred_final = evaluate_improved(
            loaded_model, test_dataloader, device, criterion
        )

    nasa = score_nasa(y_pred_final - y_true_final)    

    # =========================================================================
    #  STEP 8 — Print metrics
    # =========================================================================


    print(f"\n── Final Evaluation on Test Set ──────────────────────────")
    print(f"\n{'─'*55}")
    print(f"  Improved Model — Final Evaluation  ({eng_type})")
    print(f"{'─'*55}")
    print(f"  File        : {model_path}")
    print(f"  RMSE : {final_mets['RMSE']:.4f}")
    print(f"  MAE  : {final_mets['MAE']:.4f}")
    print(f"  R²   : {final_mets['R2']:.4f}")
    print(f"  NASA Score  : {nasa:.2f}")
    print(f"{'─'*55}")



    # =========================================================================
    #  STEP 9 — Plot predicted vs actual RUL
    # =========================================================================

    ind = np.argsort(-y_true_final)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_final[ind], color="steelblue", linewidth=1.8,
             label="Actual RUL")
    plt.plot(y_pred_final[ind], "ro-", color="tomato", markersize=3,
             linewidth=1.0, label="Predicted RUL")

    plt.title(
        f"Actual vs Predicted RUL — {eng_type}  "        
        f"RMSE={final_mets['RMSE']:.4f}  "
        f"MAE={final_mets['MAE']:.4f}  "
        f"R²={final_mets['R2']:.4f}  "
        f"NASA={nasa:.1f}",
        fontsize=11,
    )
    plt.xlabel("Index (sorted by descending true RUL)")
    plt.ylabel("Remaining Useful Life")
    plt.legend(loc="upper right", framealpha=0.7)
    plt.tight_layout()
    plt.savefig(f"rul_plot_{eng_type}.png", dpi=300)
    plt.show()


# %%
####### USE FOR EXPERIMENT 4 ONLY - Rotary Positional Embeddings (RoPE) ########
##### RE-LOAD MODEL AND DISPLAY REULTS ####
if (experiment_mode == 'Rotational Positional Encoding'):
    import torch
    import numpy as np
    import torch.nn as nn
    from improve_transformer      import evaluate_improved, ensemble_predict, score_nasa
    from improve_transformer_rope import PatchTST_RUL_RoPE_Model

    # =============================================================================
    #  STEP 1 — Load checkpoint and identify its type
    # =============================================================================

    model_path = f"BEST_rope_{eng_type}.pt"
    checkpoint  = torch.load(model_path, map_location=device)

    is_ensemble = isinstance(checkpoint, dict) and "ensemble" in checkpoint

    print(f"Checkpoint type         : {'ENSEMBLE' if is_ensemble else 'SINGLE MODEL'}")
    if is_ensemble:
        print(f"  Members               : {len(checkpoint['ensemble'])}")
        print(f"  Best scenario         : {checkpoint.get('scenario',  'unknown')}")
        print(f"  Positional encoding   : {checkpoint.get('positional_encoding', 'unknown')}")

    # =============================================================================
    #  STEP 2 — Confirm this is a RoPE checkpoint
    #  The RoPE model stores rope buffers under keys like:
    #    temporal_encoder.encoder.layers.0.self_attn.rope.inv_freq
    #  The original sinusoidal model stores:
    #    temporal_encoder.pos_enc.pe
    #  We check for these distinguishing keys before proceeding.
    # =============================================================================

    def _first_state_dict(ckpt: dict) -> dict:
        """Return the first state dict regardless of single/ensemble format."""
        return ckpt["ensemble"][0] if (isinstance(ckpt, dict) and "ensemble" in ckpt) else ckpt


    first_sd = _first_state_dict(checkpoint)

    # Key that exists only in RoPE models
    ROPE_KEY  = "temporal_encoder.encoder.layers.0.self_attn.rope.inv_freq"
    # Key that exists only in sinusoidal PE models
    SINUS_KEY = "temporal_encoder.pos_enc.pe"

    has_rope  = ROPE_KEY  in first_sd
    has_sinus = SINUS_KEY in first_sd

    if has_sinus and not has_rope:
        raise RuntimeError(
            f"The checkpoint '{model_path}' contains sinusoidal PE keys "
            f"('{SINUS_KEY}') and NO RoPE keys — it was saved by "
            f"improve_transformer.py, NOT improve_transformer_rope.py.\n"
            f"Load it with the standard loading code instead."
        )

    if not has_rope:
        print(
            "  WARNING: RoPE buffer keys not found. "
            "The model may have been saved before rope buffers were registered. "
            "Proceeding anyway — if load_state_dict fails, check the file origin."
        )
    else:
        print(f"  RoPE keys confirmed   : {ROPE_KEY} ✔")

    # =============================================================================
    #  STEP 3 — Detect architecture from weight shapes
    #  Same logic as the sinusoidal model EXCEPT:
    #    • rope.inv_freq shape gives us d_k = d_model // n_heads  (cross-check)
    #    • no 'pos_enc' keys exist
    #    • head_hidden comes from fusion_head.mlp.0.bias (same key)
    # =============================================================================

    def detect_rope_arch(sd: dict) -> dict:
        """
        Infer PatchTST_RUL_RoPE_Model hyperparameters from tensor shapes.
        Keys are identical to the sinusoidal model except 'pos_enc' is absent
        and 'rope' buffers are present.
        """
        arch = {}

        # Encoder embedding dimensions
        arch["d_model_t"]   = sd["temporal_encoder.patch_embed.proj.bias"].shape[0]
        arch["d_model_c"]   = sd["sensor_encoder.patch_embed.proj.bias"].shape[0]

        # Patch lengths
        arch["patch_len_t"] = sd["temporal_encoder.patch_embed.proj.weight"].shape[1]
        arch["patch_len_c"] = sd["sensor_encoder.patch_embed.proj.weight"].shape[1]

        # Number of encoder layers
        t_idx = {int(k.split(".")[3])
                for k in sd if k.startswith("temporal_encoder.encoder.layers.")}
        c_idx = {int(k.split(".")[3])
                for k in sd if k.startswith("sensor_encoder.transformer_encoder.layers.")}
        arch["n_layers_t"]  = max(t_idx) + 1
        arch["n_layers_c"]  = max(c_idx) + 1

        # Feedforward hidden dim
        arch["d_ff_t"] = sd["temporal_encoder.encoder.layers.0.linear1.bias"].shape[0]
        arch["d_ff_c"] = sd["sensor_encoder.transformer_encoder.layers.0.linear1.bias"].shape[0]

        # Fusion head MLP hidden dim
        arch["head_hidden"] = sd["fusion_head.mlp.0.bias"].shape[0]

        # n_heads: fixed at 8 in all your configs (not recoverable from weights alone)
        arch["n_heads_t"] = 8
        arch["n_heads_c"] = 8

        # Stride: your convention is stride_t = patch_len_t // 2, stride_c = 1
        arch["stride_t"]  = arch["patch_len_t"] // 2
        arch["stride_c"]  = 1

        # RoPE-specific: d_k cross-check from rope buffer
        # inv_freq shape = (d_k // 2,)  →  d_k = 2 * inv_freq.shape[0]
        rope_key = "temporal_encoder.encoder.layers.0.self_attn.rope.inv_freq"
        if rope_key in sd:
            d_k_from_rope = sd[rope_key].shape[0] * 2
            d_k_from_arch = arch["d_model_t"] // arch["n_heads_t"]
            if d_k_from_rope != d_k_from_arch:
                print(
                    f"  WARNING: RoPE d_k={d_k_from_rope} ≠ d_model/n_heads "
                    f"d_k={d_k_from_arch}. Check n_heads_t."
                )

        # Dropout not stored in weights — 0.0 is correct at inference
        arch["dropout_t"] = 0.0
        arch["dropout_c"] = 0.0

        return arch


    arch = detect_rope_arch(first_sd)

    print("\nDetected architecture:")
    for k, v in arch.items():
        print(f"  {k:<18} : {v}")

    # =============================================================================
    #  STEP 4 — Probe for the correct window size L
    #  SensorChannelTransformerEncoderRoPE stores self.L and asserts it matches
    #  the input — L cannot be read from weights.  We try each candidate L with
    #  a dummy forward pass; the one that succeeds is the correct L.
    # =============================================================================

    def build_rope_model(arch: dict, C: int, L: int, device,
                        max_seq_len: int = 2048,
                        rope_base:   float = 10000.0) -> nn.Module:
        """Construct PatchTST_RUL_RoPE_Model from detected arch + explicit C, L."""
        return PatchTST_RUL_RoPE_Model(
            C            = C,
            L            = L,
            d_model_t    = arch["d_model_t"],
            n_heads_t    = arch["n_heads_t"],
            n_layers_t   = arch["n_layers_t"],
            d_ff_t       = arch["d_ff_t"],
            dropout_t    = arch["dropout_t"],
            patch_len_t  = arch["patch_len_t"],
            stride_t     = arch["stride_t"],
            patch_len_c  = arch["patch_len_c"],
            stride_c     = arch["stride_c"],
            d_model_c    = arch["d_model_c"],
            n_heads_c    = arch["n_heads_c"],
            n_layers_c   = arch["n_layers_c"],
            d_ff_c       = arch["d_ff_c"],
            dropout_c    = arch["dropout_c"],
            head_hidden  = arch["head_hidden"],
            use_bn_temporal = True,
            use_bn_channel  = True,
            max_seq_len  = max_seq_len,
            rope_base    = rope_base,
        ).to(device)


    C_val      = len(features)
    LARGER_W   = max(window_size, 50)
    candidates = sorted(set([window_size, LARGER_W]))   # e.g. [40, 50]

    print(f"\nProbing window sizes {candidates} with C={C_val} ...")
    correct_L = None

    for L_try in candidates:
        try:
            probe = build_rope_model(arch, C_val, L_try, device)
            probe.load_state_dict(first_sd, strict=True)
            probe.eval()
            dummy = torch.zeros(2, C_val, L_try, device=device)
            with torch.no_grad():
                _ = probe(dummy)
            correct_L = L_try
            print(f"  L={L_try}  ✔  forward pass succeeded → using L={L_try}")
            del probe
            break
        except (AssertionError, RuntimeError) as exc:
            print(f"  L={L_try}  ✘  ({type(exc).__name__}: {str(exc)[:90]})")

    if correct_L is None:
        raise RuntimeError(
            "Could not determine the correct window size L automatically. "
            f"Tried: {candidates}. "
            "If the model was trained with a custom window, add that value "
            "to `candidates` manually and re-run."
        )

    # =============================================================================
    #  STEP 5 — Load model(s) with confirmed C and L
    # =============================================================================

    if is_ensemble:
        loaded_models = []
        for i, sd in enumerate(checkpoint["ensemble"]):
            m = build_rope_model(arch, C_val, correct_L, device)
            m.load_state_dict(sd, strict=True)
            m.eval()
            loaded_models.append(m)
            print(f"  Ensemble member {i+1} loaded ✔")
        print(f"\nEnsemble of {len(loaded_models)} RoPE models ready.")
    else:
        loaded_model = build_rope_model(arch, C_val, correct_L, device)
        loaded_model.load_state_dict(checkpoint, strict=True)
        loaded_model.eval()
        print("RoPE model loaded and set to eval mode ✔")

    # =============================================================================
    #  STEP 6 — Rebuild test loader with the correct window size
    #  If X_testf was built with a different window than correct_L, rebuild it.
    # =============================================================================

    current_Xtestf_L = X_testf.shape[1]

    if current_Xtestf_L != correct_L:
        print(f"\nX_testf was built with window={current_Xtestf_L} "
            f"but the model needs window={correct_L}.")
        print("Re-running create_testing_sequences_sw with the correct window ...")
        X_testf_for_eval = create_testing_sequences_sw(
            X_test, features, correct_L, num_of_batches
        )
        print(f"  New X_testf shape: {X_testf_for_eval.shape}")
    else:
        X_testf_for_eval = X_testf
        print(f"\nX_testf window matches model (L={correct_L}) — no rebuild needed.")

    # from MainSingleEng_FD001_F2_FD003_F4_Final import make_test_loader

    class RULWindowDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            assert X.ndim == 3, "X must be (N, C, L)"
            assert y.ndim == 1, "y must be (N,)"
            self.X = X.astype(np.float32, copy=False)
            self.y = y.astype(np.float32, copy=False)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


    def make_test_loader(X_testf, y_test, batch_size=64, num_workers=0,
                        use_cuda=torch.cuda.is_available()):
        X_test_trans = X_testf.transpose(0, 2, 1)
        test_ds = RULWindowDataset(X_test_trans, y_test)
        pin = bool(use_cuda)
        test_loader = DataLoader(
            test_ds,
            batch_size=min(batch_size, len(test_ds)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        )
        return test_loader


    test_dataloader = make_test_loader(
        X_testf     = X_testf_for_eval,
        y_test      = y_test,
        batch_size  = 64,
        num_workers = 0,
        use_cuda    = torch.cuda.is_available(),
    )

    # =============================================================================
    #  STEP 7 — Run inference
    # =============================================================================

    criterion = nn.MSELoss()

    if is_ensemble:
        y_pred_final, y_true_final, final_mets = ensemble_predict(
            loaded_models, test_dataloader, device
        )
    else:
        _, final_mets, y_true_final, y_pred_final = evaluate_improved(
            loaded_model, test_dataloader, device, criterion
        )

    nasa = score_nasa(y_pred_final - y_true_final)

    print(f"\n── RoPE Model — Final Evaluation on Test Set ──────────────")
    print(f"  Model file  : {model_path}")
    print(f"  Type        : {'Ensemble' if is_ensemble else 'Single'}")
    print(f"  L (window)  : {correct_L}")
    print(f"  C (channels): {C_val}")
    print(f"  RMSE        : {final_mets['RMSE']:.4f}")
    print(f"  MAE         : {final_mets['MAE']:.4f}")
    print(f"  R²          : {final_mets['R2']:.4f}")
    print(f"  NASA score  : {nasa:.2f}")

    # =========================================================================
    #  STEP 8 — Plot predicted vs actual RUL
    # =========================================================================

    ind = np.argsort(-y_true_final)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_final[ind], color="steelblue", linewidth=1.8,
             label="Actual RUL")
    plt.plot(y_pred_final[ind], "ro-", color="tomato", markersize=3,
             linewidth=1.0, label="Predicted RUL")

    plt.title(
        f"Actual vs Predicted RUL — {eng_type}  (ROPE)"
        f"RMSE={final_mets['RMSE']:.4f}  "
        f"MAE={final_mets['MAE']:.4f}  "
        f"R²={final_mets['R2']:.4f}  "
        f"NASA={nasa:.1f}",
        fontsize=11,
    )
    plt.xlabel("Index (sorted by descending true RUL)")
    plt.ylabel("Remaining Useful Life")
    plt.legend(loc="upper right", framealpha=0.7)
    plt.tight_layout()
    plt.savefig("rul_plot.png", dpi=300)
    plt.show()



# %%
#####################################################################################
####### USE FOR EXPERIMENT 5 ONLY - BERT ########
##### RE-LOAD MODEL AND DISPLAY REULTS ####
if (experiment_mode == 'BERT'):
    import torch
    import numpy as np
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, Dataset
    from improve_transformer      import evaluate_improved, ensemble_predict, score_nasa
    from improve_transformer_bert import PatchTST_BERT_Model

    # =========================================================================
    #  STEP 1 — Load checkpoint
    # =========================================================================

    model_path  = f"BEST_bert_{eng_type}.pt"
    checkpoint  = torch.load(model_path, map_location=device)
    is_ensemble = isinstance(checkpoint, dict) and "ensemble" in checkpoint

    print(f"Checkpoint path : {model_path}")
    print(f"Checkpoint type : {'ENSEMBLE' if is_ensemble else 'SINGLE MODEL'}")

    # Capture metadata for later window-size inference
    meta_scenario = None
    if is_ensemble:
        meta_scenario = checkpoint.get("scenario", "")
        print(f"  Members        : {len(checkpoint['ensemble'])}")
        print(f"  Best scenario  : {meta_scenario}")
        print(f"  Approach       : {checkpoint.get('approach',  'unknown')}")
        print(f"  top_k_layers   : {checkpoint.get('top_k_layers', 'unknown')}")

    # =========================================================================
    #  STEP 2 — Confirm BERT checkpoint
    # =========================================================================

    def _first_state_dict(ckpt):
        return ckpt["ensemble"][0] if (isinstance(ckpt, dict) and "ensemble" in ckpt) else ckpt

    first_sd = _first_state_dict(checkpoint)

    BERT_KEY = "fusion_head.proj_t.weight"

    if BERT_KEY not in first_sd:
        raise RuntimeError(
            f"'{model_path}' is missing the BERT-specific key '{BERT_KEY}'. "
            f"This does not look like a BERT checkpoint."
        )
    print(f"  BERT fusion head confirmed : {BERT_KEY} ✔")

    # =========================================================================
    #  STEP 3 — Detect architecture from weight shapes
    # =========================================================================

    def detect_bert_arch(sd: dict) -> dict:
        arch = {}

        arch["d_model_t"]   = sd["temporal_encoder.patch_embed.proj.bias"].shape[0]
        arch["d_model_c"]   = sd["sensor_encoder.patch_embed.proj.bias"].shape[0]
        arch["patch_len_t"] = sd["temporal_encoder.patch_embed.proj.weight"].shape[1]
        arch["patch_len_c"] = sd["sensor_encoder.patch_embed.proj.weight"].shape[1]

        t_idx = {int(k.split(".")[3])
                 for k in sd if k.startswith("temporal_encoder.encoder.layers.")}
        c_idx = {int(k.split(".")[3])
                 for k in sd if k.startswith("sensor_encoder.transformer_encoder.layers.")}
        arch["n_layers_t"]  = max(t_idx) + 1
        arch["n_layers_c"]  = max(c_idx) + 1

        arch["d_ff_t"] = sd["temporal_encoder.encoder.layers.0.linear1.bias"].shape[0]
        arch["d_ff_c"] = sd["sensor_encoder.transformer_encoder.layers.0.linear1.bias"].shape[0]

        proj_t_in           = sd[BERT_KEY].shape[1]
        arch["top_k"]       = proj_t_in // arch["d_model_t"]
        arch["d_proj"]      = sd[BERT_KEY].shape[0]
        arch["head_hidden"] = sd["fusion_head.mlp.0.bias"].shape[0]

        arch["n_heads_t"] = 8
        arch["n_heads_c"] = 8
        arch["stride_t"]  = arch["patch_len_t"] // 2
        arch["stride_c"]  = 1
        arch["dropout_t"] = 0.0
        arch["dropout_c"] = 0.0

        return arch

    arch = detect_bert_arch(first_sd)

    print("\nDetected architecture:")
    for k, v in arch.items():
        print(f"  {k:<18} : {v}")

    if is_ensemble and "top_k_layers" in checkpoint:
        meta_k = int(checkpoint["top_k_layers"])
        if meta_k != arch["top_k"]:
            print(f"  NOTE: top_k from metadata ({meta_k}) overrides "
                  f"weight-shape inference ({arch['top_k']}).")
        arch["top_k"] = meta_k

    # =========================================================================
    #  STEP 4 — Determine the correct window size L
    #
    #  Strategy:
    #    (a) If metadata names a scenario whose training used LARGER_W,
    #        prefer LARGER_W in the candidate ordering.
    #    (b) Validate each candidate by predicting on a small test sample
    #        and checking that RMSE is plausible (< 30). A wrong L produces
    #        RMSE ≈ 60+ which we reject. This catches the silent failure
    #        where the forward pass runs but on misaligned data.
    # =========================================================================

    def build_bert_model(arch, C, L, device):
        return PatchTST_BERT_Model(
            C            = C,
            L            = L,
            d_model_t    = arch["d_model_t"],
            n_heads_t    = arch["n_heads_t"],
            n_layers_t   = arch["n_layers_t"],
            d_ff_t       = arch["d_ff_t"],
            dropout_t    = arch["dropout_t"],
            patch_len_t  = arch["patch_len_t"],
            stride_t     = arch["stride_t"],
            patch_len_c  = arch["patch_len_c"],
            stride_c     = arch["stride_c"],
            d_model_c    = arch["d_model_c"],
            n_heads_c    = arch["n_heads_c"],
            n_layers_c   = arch["n_layers_c"],
            d_ff_c       = arch["d_ff_c"],
            dropout_c    = arch["dropout_c"],
            head_hidden  = arch["head_hidden"],
            use_bn_temporal = True,
            use_bn_channel  = True,
            top_k_layers    = arch["top_k"],
            d_proj          = arch["d_proj"],
            freeze_encoders = False,
        ).to(device)

    C_val    = len(features)
    LARGER_W = max(window_size, 50)

    # Order candidates: prefer LARGER_W for scenarios C, D, E (the ones that
    # used it during training). This handles your case directly.
    used_larger_w = (meta_scenario is not None and
                     any(tag in meta_scenario for tag in ("C", "D", "E", "ensemble")))

    if used_larger_w and LARGER_W != window_size:
        candidates = [LARGER_W, window_size]
        print(f"\nMetadata scenario '{meta_scenario}' used LARGER_W during "
              f"training. Probing {candidates} (LARGER_W first).")
    else:
        candidates = sorted(set([window_size, LARGER_W]))
        print(f"\nProbing window sizes {candidates}.")

    def _quick_rmse_check(model, X_eval_NCL_np, y_eval_np, n_samples=32):
        """Run model on a small sample, return RMSE. Tells us if L is right."""
        n = min(n_samples, len(y_eval_np))
        x = torch.from_numpy(X_eval_NCL_np[:n].astype(np.float32)).to(device)
        with torch.no_grad():
            preds = model(x).cpu().numpy()
        return float(np.sqrt(np.mean((preds - y_eval_np[:n]) ** 2)))

    correct_L = None
    PLAUSIBLE_RMSE = 30.0   # any sane Phase-2 result is < 15

    for L_try in candidates:
        try:
            probe = build_bert_model(arch, C_val, L_try, device)
            probe.load_state_dict(first_sd, strict=True)
            probe.eval()

            # Build test data at this L for the validation check
            if X_testf.shape[1] == L_try:
                X_eval = X_testf
            else:
                X_eval = create_testing_sequences_sw(
                    X_test, features, L_try, num_of_batches
                )
            X_eval_NCL = X_eval.transpose(0, 2, 1)   # (N, C, L)

            sample_rmse = _quick_rmse_check(probe, X_eval_NCL, y_test)
            print(f"  L={L_try}  forward OK, sample RMSE={sample_rmse:.2f}")

            if sample_rmse < PLAUSIBLE_RMSE:
                correct_L = L_try
                print(f"  → L={L_try} is the correct window size "
                      f"(RMSE plausible).")
                del probe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            else:
                print(f"  → L={L_try} REJECTED "
                      f"(sample RMSE {sample_rmse:.1f} > {PLAUSIBLE_RMSE}).")
                del probe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except (AssertionError, RuntimeError) as exc:
            print(f"  L={L_try}  ✘  ({type(exc).__name__}: {str(exc)[:100]})")

    if correct_L is None:
        raise RuntimeError(
            f"Could not determine window size L. Tried {candidates}. "
            f"All candidates either errored or produced implausibly high RMSE. "
            f"This usually means the checkpoint architecture differs from "
            f"what was assumed (e.g. patch_len, n_layers, or features list)."
        )

    # =========================================================================
    #  STEP 5 — Load model(s) with confirmed architecture
    # =========================================================================

    if is_ensemble:
        loaded_models = []
        for i, sd in enumerate(checkpoint["ensemble"]):
            m = build_bert_model(arch, C_val, correct_L, device)
            m.load_state_dict(sd, strict=True)
            m.eval()
            loaded_models.append(m)
            print(f"  Ensemble member {i+1} loaded ✔")
        print(f"\nEnsemble of {len(loaded_models)} BERT models ready.")
    else:
        loaded_model = build_bert_model(arch, C_val, correct_L, device)
        loaded_model.load_state_dict(checkpoint, strict=True)
        loaded_model.eval()
        print("BERT model loaded and set to eval mode ✔")

    # =========================================================================
    #  STEP 6 — Build the test loader EXACTLY as run_bert_scenarios does
    # =========================================================================

    class _SimpleDS(Dataset):
        def __init__(self, X, y):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i):
            return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

    if X_testf.shape[1] != correct_L:
        print(f"\nX_testf window={X_testf.shape[1]} ≠ model L={correct_L} — "
              f"rebuilding ...")
        X_testf_eval = create_testing_sequences_sw(
            X_test, features, correct_L, num_of_batches
        )
        print(f"  Rebuilt X_testf shape: {X_testf_eval.shape}  (N, L, C)")
    else:
        X_testf_eval = X_testf
        print(f"\nX_testf window matches model (L={correct_L}) — "
              f"no rebuild needed.")

    X_testf_for_model = X_testf_eval.transpose(0, 2, 1)
    print(f"  Test tensor shape fed to model: {X_testf_for_model.shape}  "
          f"(N, C, L)")

    test_dataloader = DataLoader(
        _SimpleDS(X_testf_for_model, y_test),
        batch_size = 64,
        shuffle    = False,
        num_workers= 0,
    )

    # =========================================================================
    #  STEP 7 — Run inference
    # =========================================================================

    criterion = nn.MSELoss()

    if is_ensemble:
        y_pred_final, y_true_final, final_mets = ensemble_predict(
            loaded_models, test_dataloader, device
        )
    else:
        _, final_mets, y_true_final, y_pred_final = evaluate_improved(
            loaded_model, test_dataloader, device, criterion
        )

    nasa = score_nasa(y_pred_final - y_true_final)

    # =========================================================================
    #  STEP 8 — Print metrics
    # =========================================================================

    print(f"\n{'─'*55}")
    print(f"  BERT Model — Final Evaluation  ({eng_type})")
    print(f"{'─'*55}")
    print(f"  File        : {model_path}")
    print(f"  Type        : {'Ensemble' if is_ensemble else 'Single'}")
    print(f"  Scenario    : {meta_scenario or 'n/a'}")
    print(f"  L (window)  : {correct_L}")
    print(f"  C (channels): {C_val}")
    print(f"  Top-K layers: {arch['top_k']}")
    print(f"{'─'*55}")
    print(f"  RMSE        : {final_mets['RMSE']:.4f}")
    print(f"  MAE         : {final_mets['MAE']:.4f}")
    print(f"  R²          : {final_mets['R2']:.4f}")
    print(f"  NASA Score  : {nasa:.2f}")
    print(f"{'─'*55}")

    # =========================================================================
    #  STEP 9 — Plot predicted vs actual RUL
    # =========================================================================

    ind = np.argsort(-y_true_final)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_final[ind], color="steelblue", linewidth=1.8,
             label="Actual RUL")
    plt.plot(y_pred_final[ind], "ro-", color="tomato", markersize=3,
             linewidth=1.0, label="Predicted RUL")

    plt.title(
        f"Actual vs Predicted RUL — {eng_type}  "
        f"(BERT top-{arch['top_k']} layers, L={correct_L})\n"
        f"RMSE={final_mets['RMSE']:.4f}  "
        f"MAE={final_mets['MAE']:.4f}  "
        f"R²={final_mets['R2']:.4f}  "
        f"NASA={nasa:.1f}",
        fontsize=11,
    )
    plt.xlabel("Index (sorted by descending true RUL)")
    plt.ylabel("Remaining Useful Life")
    plt.legend(loc="upper right", framealpha=0.7)
    plt.tight_layout()
    plt.savefig("rul_plot.png", dpi=300)
    plt.show()




# %%
# import matplotlib
# matplotlib.use('TkAgg', force=True)   # or 'Qt5Agg'

# import matplotlib.pyplot as plt

# import numpy as np

# plt.figure(figsize=(10, 5))
# ind = np.argsort(-y_true_final)
# plt.plot(y_true_final[ind], label='Actual RUL')
# plt.plot(y_pred_final[ind], 'ro-', label='Predicted RUL')
# # plt.plot(y_predb[ind])
# plt.title(f'Actual vs Predicted RUL for {eng_type}')
# plt.xlabel('Index')
# plt.ylabel('Remaining Useful Life')
# plt.legend(loc='upper right', framealpha=0.7)    # Top-right inside the plot
# plt.tight_layout()
# plt.show()
# plt.savefig("rul_plot.png", dpi=300)


# %%
# =============================================================================
#  HOW TO CALL FROM YOUR TRAINING NOTEBOOK
# =============================================================================
# Black Background
from encoder_fusion_experiment import run_fusion_experiment
if is_ensemble:
    fig, embs, ablation = run_fusion_experiment(
        model            = loaded_models[0],
        test_dataloader  = test_dataloader,
        device           = device,
        save_path        = "fusion_experiment_results.png",
    )
else:
    fig, embs, ablation = run_fusion_experiment(
        model            = loaded_model,
        test_dataloader  = test_dataloader,
        device           = device,
        save_path        = "fusion_experiment_results.png",
    )

fig

# =============================================================================
# White background
from encoder_fusion_experiment_whitebackground import run_fusion_experiment
if is_ensemble:
    fig, embs, ablation = run_fusion_experiment(
        model            = loaded_models[0],
        test_dataloader  = test_dataloader,
        device           = device,
        save_path        = "fusion_experiment_results.png",
    )
else:
    fig, embs, ablation = run_fusion_experiment(
        model            = loaded_model,
        test_dataloader  = test_dataloader,
        device           = device,
        save_path        = "fusion_experiment_results.png",
    )

fig

### END OF CODE 2  ############################################


# %%


# # Process to load model
# # Step 1 - Check if model is an ensemble or single model
# checkpoint = torch.load(f"BEST_improved_{eng_type}.pt", map_location=device)
# if isinstance(checkpoint, dict) and "ensemble" in checkpoint:
#     print("Ensemble —", checkpoint.get("scenario"))  # e.g. "E (ensemble)" from Scenario D base
# else:
#     print("Single model")

# # Step 2 - get the cfg object related to the desired scenario
# from improve_transformer import scenario_A_config, scenario_B_config, scenario_C_config, scenario_D_config

# desired_scenario = 'B'
# if desired_scenario == 'A':
#     cfg = scenario_A_config(features=features, window_size=window_size)
# elif desired_scenario == 'B':
#     cfg = scenario_B_config(features=features, window_size=window_size)
# elif desired_scenario == 'C':
#     cfg = scenario_C_config(features=features, window_size=50)
# elif desired_scenario == 'D':
#     cfg = scenario_B_config(features=features, window_size=50)

# # Confirm the config values
# print(f"C            : {cfg.C}")
# print(f"L            : {cfg.L}")
# print(f"d_model_t    : {cfg.d_model_t}")
# print(f"n_heads_t    : {cfg.n_heads_t}")
# print(f"n_layers_t   : {cfg.n_layers_t}")
# print(f"d_ff_t       : {cfg.d_ff_t}")
# print(f"d_model_c    : {cfg.d_model_c}")
# print(f"n_heads_c    : {cfg.n_heads_c}")
# print(f"n_layers_c   : {cfg.n_layers_c}")
# print(f"d_ff_c       : {cfg.d_ff_c}")
# print(f"head_hidden  : {cfg.head_hidden}")
# print(f"patch_len_t  : {cfg.patch_len_t}")
# print(f"stride_t     : {cfg.stride_t}")
# print(f"patch_len_c  : {cfg.patch_len_c}")
# print(f"stride_c     : {cfg.stride_c}")

# # Step 3 - Get and load model and parameters
# # ── Configuration ─────────────────────────────────────────────────────────────
# model_path = f"BEST_improved_{eng_type}.pt"
# checkpoint = torch.load(model_path, map_location=device)

# # ── Detect whether the checkpoint is a single model or an ensemble ─────────────
# is_ensemble = isinstance(checkpoint, dict) and "ensemble" in checkpoint

# if is_ensemble:
#     # ── Ensemble: rebuild one model per saved state ────────────────────────────
#     print(f"Checkpoint type : ENSEMBLE  ({len(checkpoint['ensemble'])} members)")
#     print(f"Best scenario   : {checkpoint.get('scenario', 'unknown')}")

#     loaded_models = []
#     for i, state_dict in enumerate(checkpoint["ensemble"]):
#         m = PatchTST_RUL_Model(
#             C=cfg.C, L=cfg.L,
#             d_model_t=cfg.d_model_t, n_heads_t=cfg.n_heads_t,
#             n_layers_t=cfg.n_layers_t, d_ff_t=cfg.d_ff_t,
#             dropout_t=cfg.dropout_t,
#             patch_len_t=cfg.patch_len_t, stride_t=cfg.stride_t,
#             patch_len_c=cfg.patch_len_c, stride_c=cfg.stride_c,
#             d_model_c=cfg.d_model_c, n_heads_c=cfg.n_heads_c,
#             n_layers_c=cfg.n_layers_c, d_ff_c=cfg.d_ff_c,
#             dropout_c=cfg.dropout_c,
#             head_hidden=cfg.head_hidden,
#             pooling="mean",
#             use_bn_temporal=True,
#             use_bn_channel=True,
#         ).to(device)
#         m.load_state_dict(state_dict)
#         m.eval()
#         loaded_models.append(m)
#         print(f"  Ensemble member {i+1} loaded ✔")

#     print(f"\nEnsemble of {len(loaded_models)} models ready for inference.")

# else:
#     # ── Single model: checkpoint IS the state dict ─────────────────────────────
#     print("Checkpoint type : SINGLE MODEL")

#     loaded_model = PatchTST_RUL_Model(
#         C=cfg.C, L=cfg.L,
#         d_model_t=cfg.d_model_t, n_heads_t=cfg.n_heads_t,
#         n_layers_t=cfg.n_layers_t, d_ff_t=cfg.d_ff_t,
#         dropout_t=cfg.dropout_t,
#         patch_len_t=cfg.patch_len_t, stride_t=cfg.stride_t,
#         patch_len_c=cfg.patch_len_c, stride_c=cfg.stride_c,
#         d_model_c=cfg.d_model_c, n_heads_c=cfg.n_heads_c,
#         n_layers_c=cfg.n_layers_c, d_ff_c=cfg.d_ff_c,
#         dropout_c=cfg.dropout_c,
#         head_hidden=cfg.head_hidden,
#         pooling="mean",
#         use_bn_temporal=True,
#         use_bn_channel=True,
#     ).to(device)
#     loaded_model.load_state_dict(checkpoint)
#     loaded_model.eval()
#     print("Model loaded and set to eval mode ✔")

# # ── Run inference ──────────────────────────────────────────────────────────────
# criterion = nn.MSELoss()

# # Rebuild test_dataloader from X_testf and y_test
# # These are the same arrays created earlier by create_testing_sequences_sw
# class RULWindowDataset(Dataset):
#     def __init__(self, X: np.ndarray, y: np.ndarray):
#         assert X.ndim == 3, "X must be (N, C, L)"
#         assert y.ndim == 1, "y must be (N,)"
#         self.X = X.astype(np.float32, copy=False)
#         self.y = y.astype(np.float32, copy=False)

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


# def make_test_loader(X_testf, y_test, batch_size=64, num_workers=0,
#                      use_cuda=torch.cuda.is_available()):
#     X_test_trans = X_testf.transpose(0, 2, 1)
#     test_ds = RULWindowDataset(X_test_trans, y_test)
#     pin = bool(use_cuda)
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=min(batch_size, len(test_ds)),
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin,
#     )
#     return test_loader



# test_dataloader = make_test_loader(
#     X_testf    = X_testf,
#     y_test     = y_test,
#     batch_size = 64,
#     num_workers = 0,
#     use_cuda   = torch.cuda.is_available(),
# )

# print(f"test_dataloader ready — {len(test_dataloader.dataset)} samples, "
#       f"{len(test_dataloader)} batches")

# if is_ensemble:
#     from improve_transformer import ensemble_predict
#     y_pred_final, y_true_final, final_mets = ensemble_predict(
#         loaded_models, test_dataloader, device
#     )
# else:
#     from improve_transformer import evaluate_improved
#     _, final_mets, y_true_final, y_pred_final = evaluate_improved(
#         loaded_model, test_dataloader, device, criterion
#     )

# print(f"\n── Final Evaluation on Test Set ──────────────────────────")
# print(f"  RMSE : {final_mets['RMSE']:.4f}")
# print(f"  MAE  : {final_mets['MAE']:.4f}")
# print(f"  R²   : {final_mets['R2']:.4f}")


# # %%

# # =============================================================================
# #  HOW TO CALL FROM YOUR TRAINING NOTEBOOK
# # =============================================================================

# from encoder_fusion_experiment import run_fusion_experiment

# if is_ensemble:
#     fig, embs, ablation = run_fusion_experiment(
#         model            = loaded_models[0],
#         test_dataloader  = test_dataloader,
#         device           = device,
#         save_path        = "fusion_experiment_results.png",
#     )
# else:
#     fig, embs, ablation = run_fusion_experiment(
#         model            = loaded_model,
#         test_dataloader  = test_dataloader,
#         device           = device,
#         save_path        = "fusion_experiment_results.png",
#     )

# fig

# # %%
