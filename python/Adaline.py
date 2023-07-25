import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def standardization(input_data):

	input_data_std = np.copy(input_data).astype(float)

	for i in range(len(input_data[0])):


		input_data_std[:,i] = ( input_data[:,i] - input_data[:,i].mean() ) / input_data[:,i].std()


	return input_data_std


class AdalineGD:

	def __init__(self, learning_rate=0.001, epoch=100, random_state=1):

		self.learning_rate = learning_rate
		self.epoch = epoch
		self.random_state = random_state

	def fit(self, input_data, result_data):

		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal( loc=0.0, scale=0.01, size=1+input_data.shape[1])
		self.cost_ = []

		for i in range(self.epoch):

			net_input = self.net_input(input_data)
			output_data = self.activation(net_input)

			error = result_data - output_data

			self.w_[1:] += self.learning_rate * input_data.T.dot(error) # update weight value
			self.w_[0] += self.learning_rate * error.sum() # update bias value

			cost = (error**2).sum() / 2.0

			self.cost_.append(cost)

		return self

	def plot_trainning(self):

		plt.plot(self.cost_, marker='o')
		plt.ylabel('error')
		plt.xlabel('epoch')
		plt.show()

	def net_input(self, input_data):

		return np.dot(input_data, self.w_[1:]) + self.w_[0]

	def activation(self, x):

		return x

	def predict(self, input_data):

		return self.activation(self.net_input(input_data))

if __name__ == '__main__':

	input_data = np.array([[1, 2, 3], [3, 4, 3], [4, 6, 3], [1, 4, 5]])
	result_data = np.array([3, 7, 10, 5])

	model = AdalineGD(learning_rate=0.01, epoch=100, random_state=1)

	input_data = standardization(input_data)

	model.fit(input_data, result_data)
	model.plot_trainning()

	print(model.predict([2,8, 7]))

