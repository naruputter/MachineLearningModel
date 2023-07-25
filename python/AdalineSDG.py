import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## AdalineSDG add partial to train by didnt set new loop

def standardization(input_data):

	input_data_std = np.copy(input_data).astype(float)

	for i in range(len(input_data[0])):


		input_data_std[:,i] = ( input_data[:,i] - input_data[:,i].mean() ) / input_data[:,i].std()


	return input_data_std

class AdalineSDG:

	def __init__(self, learning_rate=0.01, epoch=100, shuffle=True, random_state=None):

		self.learning_rate = learning_rate
		self.epoch = epoch
		self.w_init = False
		self.shuffle = shuffle
		self.random_state = random_state

	def initialize_weight(self, m):

		self.rgen = np.random.RandomState(self.random_state)
		self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)

		self.w_init = True

	def shuffle_data(self, input_data, result_data):

		r = self.rgen.permutation(len(result_data))

		return input_data[r], result_data[r]

	def activation(self, x):

		return x

	def net_input(self, input_data):

		return np.dot(input_data, self.w_[1:]) + self.w_[0]

	def update_weight(self, in_data, res_data):

		out_data = self.activation(self.net_input(in_data))

		error = res_data - out_data

		self.w_[1:] += self.learning_rate * in_data.dot(error)
		self.w_[0] += self.learning_rate * error

		cost = 0.5 * error**2

		return cost


	def fit(self, input_data, result_data):

		self.initialize_weight(input_data.shape[1])
		self.cost_ = []

		for i in range(self.epoch):

			if self.shuffle :

				input_data, result_data = self.shuffle_data(input_data, result_data)

			cost = []

			for in_data, res_data in zip(input_data, result_data):

				cost.append(self.update_weight(in_data, res_data))

			avg_cost = sum(cost) / len(result_data)
			self.cost_.append(avg_cost)


		return self

	def partial_fit(self, input_data, result_data):

		if not self.w_init :

			self.initialize_weight(input_data.shape[1])

		if result_data.ravel().shape[0] > 1 :

			for in_data, res_data in zip(input_data, result_data):

				self.update_weight(in_data, res_data)

		else:

			self.update_weight(input_data, result_data)

		return self

	def plot_trainning(self):

		plt.plot(self.cost_, marker='o')
		plt.ylabel('error')
		plt.xlabel('epoch')
		plt.show()



if __name__ == '__main__':

	input_data = np.array([[1, 2], [3, 4], [4, 6], [1, 9]])
	result_data = np.array([3, 7, 10, 5])
	
	input_data = standardization(input_data)

	model = AdalineSDG()
	model.fit(input_data, result_data)

	model.plot_trainning()

	print('---------------')

	new_in_data = np.array([[2, 4]])
	new_res_data = np.array(6)

	model.partial_fit(input_data, result_data)






