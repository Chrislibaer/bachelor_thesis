from pandas_datareader import data
import matplotlib.pyplot as plt
from cnnmodel import CNNModel
from datasplitter import split_data
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.optim as optim
import numpy as np
import random
import pandas as pd


TRAIN_SPLIT = 5000 # Half of the data is used for hyperparameter tuning, the second half is used for out of sample testing
INPUT_DIMS = 100 # Input length of the model in days
OUTPUT_DIMS = 1 # Output length of the model in days
STEP_SIZE = 1000 # step size for the forward testing: 
# the model is trained on the first STEP_SIZE days,
# then predicts the next STEP_SIZE days and is then trained
# on the first 2*STEP_SIZE days to predict the next STEP_SIZE days
# until the end is reached

synthetic_data = np.sin([i for i in np.arange(0, 1_00, 0.01)])
rnd_inds = random.choices(range(10_000), k=20)
for rnd in rnd_inds:
    synthetic_data[rnd] = random.uniform(-2, 2)

# For Training
#synthetic_data = synthetic_data[:TRAIN_SPLIT]

current_ind = STEP_SIZE
first_split_train = STEP_SIZE
prediction_error_combined = []

split_date = TRAIN_SPLIT
text_left_date = 5000 - 1200
text_right_date = 5000 + 100

while current_ind < 10000 - STEP_SIZE + 1:
    split_train = current_ind
    split_test = current_ind + STEP_SIZE

    x_train, x_test, y_train, y_test = split_data(synthetic_data, input_dims=INPUT_DIMS, output_dims=OUTPUT_DIMS, split=(split_train, split_test))


    train_data = torch.utils.data.TensorDataset(Tensor(x_train), Tensor(y_train))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True, num_workers=0)


    model = CNNModel(input_dims=INPUT_DIMS, output_dims=OUTPUT_DIMS)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    loss_history = []
    stop_training = False

    for ep in range(5):
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
            if i % 20 == 19: 
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


# find anomalies. anomalous is defined as the current prediction error being greater than the rolling window mean + 2.7 standard deviations
pred_error_upper = pd.Series(prediction_error_combined) - (4.4*pd.Series(prediction_error_combined).rolling(250).std() + pd.Series(prediction_error_combined).rolling(250).mean())
anomalies = np.array(range(len(synthetic_data)))[STEP_SIZE+INPUT_DIMS:][((pred_error_upper > 0)).values]

# draw results
fig, axs = plt.subplots(2, figsize=(10, 5))

axs[0].plot(synthetic_data)
axs[0].scatter(pd.Series(anomalies), synthetic_data[anomalies], c='red')
axs[0].vlines(split_date, synthetic_data.min(), synthetic_data.max())
axs[0].text(text_left_date, synthetic_data.max()-0.2, 'training set', fontsize='medium')
axs[0].text(text_right_date, synthetic_data.max()-0.2, 'test set', fontsize='medium')

axs[1].plot(range(len(synthetic_data)), [0 for _ in range(STEP_SIZE + INPUT_DIMS)]+prediction_error_combined)
axs[1].scatter(pd.Series(anomalies), np.array([0 for _ in range(STEP_SIZE+INPUT_DIMS)]+prediction_error_combined)[anomalies], c='red')
axs[1].vlines(split_date, min(prediction_error_combined), max(prediction_error_combined))
axs[1].text(text_left_date, max(prediction_error_combined)-0.2, 'training set', fontsize='medium')
axs[1].text(text_right_date, max(prediction_error_combined)-0.2, 'test set', fontsize='medium')

plt.show()
fig.savefig('synthetic_cnn.pdf')