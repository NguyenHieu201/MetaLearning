import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# get data in time range
def get_data(path, start_day, finish_day):
    df = pd.read_csv(path)
    df_result = df[(df['datetime'] > start_day) & (df['datetime'] < finish_day)]
    return df_result.close.to_numpy()

def time_series_processing(data, mode, setting):
    if mode == "trend-ratio":
        n_sample = data.shape[0]
        seq_len = setting["seq-len"]
        future = setting["future"]
        x = [data[i : i+seq_len] for i in range(0, n_sample - seq_len)]
        y = [data[i] for i in range(seq_len + future - 1, n_sample)]
        temp = [data[i+seq_len - 1] for i in range(0, n_sample - seq_len)]
        n_sample = min(len(x), len(y))
        x = x[:n_sample]
        y = y[:n_sample]
        temp = temp[:n_sample]
        y = np.array(y, dtype=np.float32) / np.array(temp, dtype=np.float32)
        return {
            "X": np.array(x, dtype=np.float32),
            "Y": np.array(y, dtype=np.float32).reshape(-1, 1)
        }
        

# Preprocessing data
def preprocessing(path, name, prediction_mode, setting, time):
    start_day = time["start-day"]
    finish_day = time["finish-day"]
    data = get_data(path, start_day, finish_day)
    data = time_series_processing(data, prediction_mode, setting)
    domain = data_scale(name, data)
    return domain
    
# Scaler data for test
def data_scale(name, data):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(data["X"])
    y_scaler.fit(data["Y"])
    
    # Transform
    data["X"] = x_scaler.transform(data["X"])
    data["Y"] = y_scaler.transform(data["Y"])

    return {
        "name": name,
        "data": data,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler
    }
    
    