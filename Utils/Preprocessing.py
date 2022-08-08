import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# get data from csv file in time range
def get_data(path, start_day, finish_day):
    df = pd.read_csv(path)
    df = df[(df['datetime'] > start_day) & (df['datetime'] < finish_day)]
    data = df.close.to_numpy()
    return data.reshape(-1, 1)

# time series processing
def time_series_processing(data, mode, setting):
    if mode == "trend-ratio":
        seq_len = setting["seq-len"]
        future = setting["future"]
        n_sample = data.shape[0]
        x = [data[i : i+seq_len] for i in range(0, n_sample - seq_len + 1)]
        y = [data[i + seq_len + future - 1] for i in range(0, n_sample - seq_len - future + 1)]
        x_last = [data[i + seq_len - 1] for i in range(0, n_sample + 1 - seq_len)]
        n_sample = min(len(x), len(y))
        x = x[:n_sample]
        y = y[:n_sample]
        x_last = x_last[:n_sample]
        x = np.array(x)
        y = np.array(y)
        x_last = np.array(x_last)

        y = (y - x_last) / x_last
        return {
            "X": x,
            "Y": y
        }

# preprocessing
def preprocessing(path, name, time, setting):
    start_day = time["start-day"]
    finish_day = time["finish-day"]
    data = get_data(path, start_day, finish_day)
    scaler = MinMaxScaler(feature_range=(0.1, 1))
    scaler.fit(data)
    data = scaler.transform(data)
    
    mode = setting["predict-mode"]
    config = setting["preprocess-setting"]
    data = time_series_processing(data, mode, config)
    
    return {
        "name": name,
        "scaler": scaler,
        "data": data
    }