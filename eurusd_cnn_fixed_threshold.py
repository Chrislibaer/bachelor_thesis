import matplotlib.pyplot as plt
from cnnmodel import CNNModel
from datasplitter import split_data
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.optim as optim
import numpy as np
import pandas as pd

torch.manual_seed(0)

TRAIN_SPLIT = 2806 # Half of the data is used for hyperparameter tuning, the second half is used for out of sample testing
INPUT_DIMS = 100 # Input length of the model in days
OUTPUT_DIMS = 1 # Output length of the model in days
STEP_SIZE = 250 # step size for the forward testing: 
# the model is trained on the first STEP_SIZE days,
# then predicts the next STEP_SIZE days and is then trained
# on the first 2*STEP_SIZE days to predict the next STEP_SIZE days
# until the end is reached

eurusd = pd.read_csv('eurusd.csv', index_col='date')
eurusd.index = pd.to_datetime(eurusd.index)

# For Training. Only use the first half of the data to optimize hyperparameters
#eurusd = eurusd.iloc[:TRAIN_SPLIT]
#print(eurusd.index[-1])
split_date = eurusd.index[TRAIN_SPLIT]
text_left_date = eurusd.index[2806 - 600]
text_right_date = eurusd.index[2806 + 100]

# only looking at the mid close price
eurusd = (eurusd['bidclose'] + eurusd['askclose']) / 2
dates = eurusd.index


current_ind = STEP_SIZE
first_split_train = STEP_SIZE
prediction_error_combined = []

while current_ind < len(eurusd) - OUTPUT_DIMS:
    split_train = current_ind
    split_test = current_ind + STEP_SIZE

    x_train, x_test, y_train, y_test = split_data(eurusd.to_numpy(), input_dims=INPUT_DIMS, output_dims=OUTPUT_DIMS, split=(split_train, split_test))

    train_data = torch.utils.data.TensorDataset(Tensor(x_train), Tensor(y_train))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True, num_workers=0)

    model = CNNModel(input_dims=INPUT_DIMS, output_dims=OUTPUT_DIMS)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    loss_history = []
    stop_training = False

    for ep in range(20):
        if stop_training:
            break
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            if stop_training:
                break
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs[:, 0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # print stats
            running_loss += loss.item()
            if i % 10 == 9:
                loss_history.append(running_loss)
                print('[%d] loss: %.6f' %
                      (ep + 1, running_loss / 2000))
                running_loss = 0.0
            if len(loss_history) > 5 and loss_history[-1] >= loss_history[-5]:
                print("loss not decreasing anymore. stopping training")
                stop_training = True

    model.eval()

    test_data = torch.utils.data.TensorDataset(Tensor(x_test), Tensor(y_test))
    testloader = torch.utils.data.DataLoader(test_data, shuffle=False)

    pred_test = []
    for i, data in enumerate(testloader): 
    	inputs, labels = data

    	pred_test.append(model(inputs).detach().numpy()[0][0])


    prediction_error = np.abs(y_test - pred_test)

    prediction_error_combined += prediction_error.tolist()

    current_ind += STEP_SIZE


anomalies = []
anomaly_pred_error_ids = []

# find anomalies. anomalous is defined as the current prediction error being greater than the rolling window mean + 2.7 standard deviations
pred_error_upper = pd.Series(prediction_error_combined)
anomalies = dates[STEP_SIZE+INPUT_DIMS:][((pred_error_upper > 0.2)).values]
anomaly_pred_error_ids = [dates.get_loc(i)-STEP_SIZE-INPUT_DIMS for i in anomalies]

# draw results
fig, axs = plt.subplots(2, figsize=(12, 6))

axs[0].plot(eurusd.iloc[STEP_SIZE+INPUT_DIMS:])
axs[0].scatter(pd.Series(anomalies), eurusd.loc[anomalies], c='red')
axs[0].vlines(split_date, 0.95, 1.62)
axs[0].text(text_left_date, 1.56, 'training set', fontsize='medium')
axs[0].text(text_right_date, 1.56, 'test set', fontsize='medium')

axs[1].plot(dates[first_split_train+INPUT_DIMS:], prediction_error_combined)
axs[1].scatter(pd.Series(anomalies), np.array(prediction_error_combined)[anomaly_pred_error_ids], c='red')
axs[1].vlines(split_date, 0.0, 0.4)
axs[1].text(text_left_date, 0.37, 'training set', fontsize='medium')
axs[1].text(text_right_date, 0.37, 'test set', fontsize='medium')

plt.show()
fig.savefig('eurusd_cnn_fixed_threshold.pdf')