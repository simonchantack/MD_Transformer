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
# Train and predict multiple scenarios using cross attention fusion
from patchtst_crossattn import run_crossattn_scenarios

leaderboard = run_crossattn_scenarios(
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

import torch
import numpy as np
import torch.nn as nn
from improve_transformer import (
    scenario_A_config, scenario_B_config,
    scenario_C_config, scenario_D_config,
    evaluate_improved, ensemble_predict, score_nasa,
)
from patchtst_crossattn import PatchTST_CrossAttn_Model

# =============================================================================
#  STEP 1 — Load checkpoint
# =============================================================================

model_path = f"BEST_crossattn_{eng_type}.pt"
checkpoint  = torch.load(model_path, map_location=device)
is_ensemble = isinstance(checkpoint, dict) and "ensemble" in checkpoint

print(f"Checkpoint type : {'ENSEMBLE' if is_ensemble else 'SINGLE MODEL'}")
if is_ensemble:
    print(f"  Members        : {len(checkpoint['ensemble'])}")
    print(f"  Best scenario  : {checkpoint.get('scenario', 'unknown')}")

# =============================================================================
#  STEP 2 — Detect architecture from weight shapes
# =============================================================================

def detect_arch(sd: dict) -> dict:
    arch = {}
    arch["d_model_t"]   = sd["temporal_encoder.patch_embed.proj.bias"].shape[0]
    arch["d_model_c"]   = sd["sensor_encoder.patch_embed.proj.bias"].shape[0]
    arch["patch_len_t"] = sd["temporal_encoder.patch_embed.proj.weight"].shape[1]
    arch["patch_len_c"] = sd["sensor_encoder.patch_embed.proj.weight"].shape[1]
    arch["d_ff_t"]      = sd["temporal_encoder.encoder.layers.0.linear1.bias"].shape[0]
    arch["d_ff_c"]      = sd["sensor_encoder.transformer_encoder.layers.0.linear1.bias"].shape[0]

    t_idx = {int(k.split(".")[3]) for k in sd if k.startswith("temporal_encoder.encoder.layers.")}
    c_idx = {int(k.split(".")[3]) for k in sd if k.startswith("sensor_encoder.transformer_encoder.layers.")}
    arch["n_layers_t"]    = max(t_idx) + 1
    arch["n_layers_c"]    = max(c_idx) + 1
    arch["n_heads_t"]     = 8
    arch["n_heads_c"]     = 8
    arch["d_model_xattn"] = sd["fusion_head.t_to_c.q_proj.weight"].shape[0]
    arch["n_heads_xattn"] = 8
    ffn_key = "fusion_head.t_to_c.ffn.0.weight"
    arch["d_ff_xattn"]    = sd[ffn_key].shape[0] if ffn_key in sd else 0
    arch["head_hidden"]   = sd["fusion_head.mlp.1.bias"].shape[0]
    arch["stride_t"]      = arch["patch_len_t"] // 2
    arch["stride_c"]      = 1
    arch["dropout_t"]     = 0.0
    arch["dropout_c"]     = 0.0
    arch["dropout_xattn"] = 0.0
    return arch


first_sd = checkpoint["ensemble"][0] if is_ensemble else checkpoint
arch      = detect_arch(first_sd)

print("\nDetected architecture:")
for k, v in arch.items():
    print(f"  {k:<20} : {v}")

# =============================================================================
#  STEP 3 — Probe for the correct window size L
#
#  SensorChannelTransformerEncoder stores self.L at construction and asserts
#  it matches the input — so we cannot know L from weights alone.
#  Strategy: try each candidate L with a zero-tensor dummy forward pass.
#  Whichever does NOT raise AssertionError is the correct L.
# =============================================================================

def build_model(arch: dict, C: int, L: int, device) -> nn.Module:
    """Construct PatchTST_CrossAttn_Model from detected arch + explicit C and L."""
    return PatchTST_CrossAttn_Model(
        C              = C,
        L              = L,
        d_model_t      = arch["d_model_t"],
        n_heads_t      = arch["n_heads_t"],
        n_layers_t     = arch["n_layers_t"],
        d_ff_t         = arch["d_ff_t"],
        dropout_t      = arch["dropout_t"],
        patch_len_t    = arch["patch_len_t"],
        stride_t       = arch["stride_t"],
        patch_len_c    = arch["patch_len_c"],
        stride_c       = arch["stride_c"],
        d_model_c      = arch["d_model_c"],
        n_heads_c      = arch["n_heads_c"],
        n_layers_c     = arch["n_layers_c"],
        d_ff_c         = arch["d_ff_c"],
        dropout_c      = arch["dropout_c"],
        head_hidden    = arch["head_hidden"],
        d_model_xattn  = arch["d_model_xattn"],
        n_heads_xattn  = arch["n_heads_xattn"],
        d_ff_xattn     = arch["d_ff_xattn"],
        dropout_xattn  = arch["dropout_xattn"],
        use_bn_temporal= True,
        use_bn_channel = True,
    ).to(device)


C_val      = len(features)
LARGER_W   = max(window_size, 50)
candidates = sorted(set([window_size, LARGER_W]))   # e.g. [40, 50]

correct_L  = None
for L_try in candidates:
    try:
        probe = build_model(arch, C_val, L_try, device)
        probe.load_state_dict(first_sd)
        probe.eval()
        dummy = torch.zeros(2, C_val, L_try, device=device)
        with torch.no_grad():
            probe(dummy)
        correct_L = L_try
        print(f"\nProbed L={L_try} ✔  — model runs without error → using L={L_try}")
        del probe
        break
    except (AssertionError, RuntimeError) as e:
        print(f"Probed L={L_try} ✘  ({type(e).__name__}: {str(e)[:80]})")

if correct_L is None:
    raise RuntimeError(
        "Could not determine the correct window size L automatically. "
        f"Tried: {candidates}. "
        "If the model was trained with a custom window size not in this list, "
        "add it manually to `candidates` above."
    )

# =============================================================================
#  STEP 4 — Rebuild config with confirmed C and L
# =============================================================================

# Pick the matching scenario config (sets lr, epochs, etc. — not critical
# for inference, but keeps the config object consistent).
if arch["d_model_t"] == 96 and arch["n_layers_t"] == 3:
    cfg = (scenario_D_config if correct_L == LARGER_W
           else scenario_B_config)(features, correct_L)
else:
    cfg = (scenario_C_config if correct_L == LARGER_W
           else scenario_A_config)(features, correct_L)

cfg.C = C_val
cfg.L = correct_L

print(f"\nFinal cfg.C={cfg.C}  cfg.L={cfg.L}  "
      f"d_model_t={cfg.d_model_t}  n_layers_t={cfg.n_layers_t}")

# =============================================================================
#  STEP 5 — Load model(s) with confirmed L
# =============================================================================

if is_ensemble:
    loaded_models = []
    for i, sd in enumerate(checkpoint["ensemble"]):
        m = build_model(arch, C_val, correct_L, device)
        m.load_state_dict(sd)
        m.eval()
        loaded_models.append(m)
        print(f"  Ensemble member {i+1} loaded ✔")
    print(f"\nEnsemble of {len(loaded_models)} models ready.")
else:
    loaded_model = build_model(arch, C_val, correct_L, device)
    loaded_model.load_state_dict(checkpoint)
    loaded_model.eval()
    print("Model loaded ✔")

# =============================================================================
#  STEP 6 — Rebuild test loader with the correct window size
#
#  If correct_L differs from the window used to build X_testf, we must
#  recreate X_testf.  X_testf.shape[1] tells us which window it was built with.
# =============================================================================

current_Xtestf_L = X_testf.shape[1]   # window used when X_testf was built

if current_Xtestf_L != correct_L:
    print(f"\nX_testf was built with window={current_Xtestf_L} "
          f"but model needs window={correct_L}.")
    print("Re-creating X_testf with the correct window size ...")
    X_testf_for_eval = create_testing_sequences_sw(
        X_test, features, correct_L, num_of_batches
    )
    print(f"New X_testf shape: {X_testf_for_eval.shape}")
else:
    X_testf_for_eval = X_testf
    print(f"\nX_testf window matches model (L={correct_L}) — no rebuild needed.")

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
#  STEP 7 — Inference
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

save_path = f"BEST_CrossAttn_{eng_type}.pt"
print(f"\n── Final Evaluation ({'Ensemble' if is_ensemble else 'Single'}) ──")
print(f"  RMSE  : {final_mets['RMSE']:.4f}")
print(f"  MAE   : {final_mets['MAE']:.4f}")
print(f"  R²    : {final_mets['R2']:.4f}")
print(f"  NASA  : {nasa:.2f}")
print(f"\n  Best RL model saved -> {save_path}")


# ---- Actual vs Predicted RUL plot ---------------------------------------
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
    f"NASA={nasa:.2f}",
    fontsize=11,
)
plt.xlabel("Index (sorted by descending true RUL)")
plt.ylabel("Remaining Useful Life")
plt.legend(loc="upper right", framealpha=0.7)
plt.tight_layout()
plt.savefig(f"rul_plot_CrossAttn_{eng_type}.png", dpi=300)
plt.show()


# %%
# =============================================================================
#  HOW TO CALL ENCODER FUSION EXPERIMENT 
# =============================================================================
# Black background
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

# %%
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

# %%