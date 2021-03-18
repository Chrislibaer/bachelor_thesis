import numpy as np


def split_data(data, input_dims=10, output_dims=1, split=(100, 100)):
	x, y = [], []
	for i in range(input_dims, len(data)):
		x.append(data[i-input_dims:i])
		y.append(data[i])

	split_train = split[0]
	split_test = split[1]

	x_train, x_test = x[:split_train], x[split_train:split_test]
	y_train, y_test = y[:split_train], y[split_train:split_test]

	return np.array(x_train), np.array(x_test),\
			 np.array(y_train), np.array(y_test)