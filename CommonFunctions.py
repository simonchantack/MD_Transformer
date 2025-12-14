
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import root_mean_squared_error , r2_score 

# Function to prepare tarininig data - by creating sequences of data in batches, each batch based on a sliding window of window_size 
def create_training_sequences_sw(df, features, window_size):
    X, y = [], []
    for engine_id in df['engine'].unique():
        engine_data = df[df['engine'] == engine_id]
        for i in range(len(engine_data) - window_size):
            X.append(engine_data[features].iloc[i:i+window_size].values)
            y.append(engine_data['rul'].iloc[i+window_size])
    return np.array(X), np.array(y)


# Function to prepare testing data - needs to have similar shape as training data
def create_testing_sequences_sw(df, features, window_size, num_of_batches):
    factor = 1
    prep_test_data = []
    for engine_id in df['engine'].unique():
        engine_data = df[df['engine'] == engine_id]
        for i in range(num_of_batches):                
            start_pt = ((len(engine_data) - 1)- (factor * window_size) - (num_of_batches - 1)) + i
            end_pt = start_pt + window_size

            # If window_size is bigger than engine_data get engine_data and 
            # pad the window with the last item in the batch
            if (len(engine_data) < window_size):
                new_start_pt = 0
                new_end_pt = len(engine_data)
                data_part1 = engine_data[features].iloc[new_start_pt:new_end_pt, :]
                # Padding data
                filler = (window_size) - len(engine_data)                 
                last_row = engine_data[features].iloc[[-1]] 
                data_filler = pd.concat([last_row] * filler, ignore_index=True)
                syntec_data = pd.concat([data_part1, data_filler])
                prep_test_data.append(syntec_data)               

            elif (start_pt < 0):   
                batch_data = engine_data[features].iloc[0:window_size, :]              
                # prep_test_data.append(engine_data[features].iloc[start_pt:end_pt, :])
                prep_test_data.append(batch_data)

            else:
                batch_data = engine_data[features].iloc[start_pt:end_pt, :]  
                prep_test_data.append(batch_data) 

    return np.array(prep_test_data)   


# Sliding/ Expanding window for training data - this is used for training data preparation
# This function creates a list of sequences for each engine, where each sequence is obtained from a sliding/expanding window of the data.
def create_training_sequences_exp(df, min_window_size):
    X = []
    for engine_id in df['engine'].unique():
        engine_data = df[df['engine'] == engine_id]
        engine_data.reset_index(drop=True, inplace=True)
        engine_data_len = len(engine_data)
        engine_max_len = engine_data_len - min_window_size + 1
        st_pt = 0
        while st_pt < engine_max_len:
            X.append(engine_data.iloc[:st_pt+min_window_size, 1:])
            st_pt += 1
    return X

# Function to prepare testing data - this is used for testing data preparation
# This function creates a list of sequences for each engine, where each sequence is obtained from a sliding/expanding window of the data.
def create_testing_sequences_exp(df):
    X = []
    for engine_id in df['engine'].unique():
        engine_data = df[df['engine'] == engine_id]
        engine_data_arr = engine_data.iloc[:,1:].to_numpy()
        X.append(engine_data_arr)        
    return X

# Prepare trarget values using sliding/expanding window
def yPrep(data, min_window_size):
    y_lst = []
    for engine_id in data['engine'].unique():
        engine_data = data[data['engine'] == engine_id]
        y_temp = engine_data['rul'][min_window_size-1:].values
        y_lst.append(y_temp)
    return np.concatenate(y_lst, axis=0)



# def create_training_sequences_expwindow(df, features, min_window_size):
#     X = []
#     for engine_id in df['engine'].unique():
#         engine_data = df[df['engine']==engine_id]
#         for i in range(len(engine_data) - min_window_size + 1):
#             exp_win_end = i + min_window_size
#             X.append(engine_data[features].iloc[:exp_win_end].values)
#     return np.array(X)

# Metrics function - to compute RMSE values
def metrics(y_true , y_pred , label = 'train'):
    '''evaluate model , by taking y_true , y_pred and label of dataset'''
    
    rmse = root_mean_squared_error(y_true , y_pred)
    r2 = r2_score(y_true , y_pred)
    print(f'for {label} set , RMSE = {rmse:0.2f} , r2_score = {r2*100:0.2f}%')


# Plot model training performance
def plot_model_hist(history):
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Plot actual and predicted
def plot_act_pred(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='orange')
    plt.title('Actual vs LSTM Predicted RUL')
    plt.xlabel('Sample Index')
    plt.ylabel('Remaining Useful Life')
    plt.legend()
    plt.grid()
    plt.show()


# Plot Predicted vs Actual RUL
def plot_act_vs_pred(y_test, y_pred):
    sns.scatterplot(x = y_test['RUL'] , y = y_pred.flatten()  , s = 30 , alpha = 0.5)

    sns.lineplot( x = [ min(y_test['RUL'])  , max(y_test['RUL']) ] , 
                y = [min(y_test['RUL']) , max(y_test['RUL'])] , color = 'red')
    

    plt.xlabel('LSTM Predicted')
    plt.ylabel('Actual')