from pandas_datareader import data
import matplotlib.pyplot as plt
from datasplitter import split_data
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.optim as optim
import time
import numpy as np
import fxcmpy
from sklearn.svm import OneClassSVM
import pandas as pd


TRAIN_SPLIT = 2806 # Half of the data is used for hyperparameter tuning, the second half is used for out of sample testing
INPUT_DIMS = 100 # Input length of the model in days
OUTPUT_DIMS = 1 # Output length of the model in days
STEP_SIZE = 250


eurusd = pd.read_csv('eurusd.csv', index_col='date')
eurusd.index = pd.to_datetime(eurusd.index)
eurusd = ((eurusd['bidclose'] + eurusd['askclose']) / 2)
dates = eurusd.index

split_date = eurusd.index[TRAIN_SPLIT]
text_left_date = eurusd.index[2806 - 600]
text_right_date = eurusd.index[2806 + 100]

current_ind = STEP_SIZE
first_split_train = STEP_SIZE
prediction_error_combined = []

while current_ind < len(eurusd) - OUTPUT_DIMS:
    split_train = current_ind
    split_test = current_ind + STEP_SIZE

    x_train, x_test, y_train, y_test = split_data(eurusd.to_numpy(), input_dims=INPUT_DIMS, output_dims=OUTPUT_DIMS, split=(split_train, split_test))

    model = OneClassSVM(nu=0.2).fit(x_train)


    print('Finished Training')
    if len(x_test) > 0:
        prediction_error_combined += list(model.predict(x_test))

    current_ind += STEP_SIZE


anomalies = dates[STEP_SIZE+INPUT_DIMS:][(pd.Series(prediction_error_combined) == -1).values]
anomaly_pred_error_ids = [dates.get_loc(i)-STEP_SIZE-INPUT_DIMS for i in anomalies]

"""fig, axs = plt.subplots(2)

axs[0].plot(eurusd)
axs[0].scatter(pd.Series(anomalies), eurusd.loc[anomalies], c='red')
axs[1].scatter(dates[first_split_train+INPUT_DIMS:-12], prediction_error_combined, s=3)
plt.show()
"""
# draw results
fig, axs = plt.subplots(2, figsize=(12, 6))

axs[0].plot(eurusd.iloc[STEP_SIZE+INPUT_DIMS:])
axs[0].scatter(pd.Series(anomalies), eurusd.loc[anomalies], c='red')
axs[0].vlines(split_date, 0.95, 1.62)
axs[0].text(text_left_date, 1.56, 'training set', fontsize='medium')
axs[0].text(text_right_date, 1.56, 'test set', fontsize='medium')

axs[1].scatter(dates[first_split_train+INPUT_DIMS:], prediction_error_combined, s=3)
axs[1].scatter(pd.Series(anomalies), np.array(prediction_error_combined)[anomaly_pred_error_ids], c='red', s=3)
axs[1].vlines(split_date, -1, 1.2)
axs[1].text(text_left_date, 1.1, 'training set', fontsize='medium')
axs[1].text(text_right_date, 1.1, 'test set', fontsize='medium')

plt.show()
fig.savefig('eurusd_one_class.pdf')
