import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

def show_image(array_image):

	plt.figure()
	plt.imshow(array_image)
	plt.colorbar()
	plt.grid(False)
	plt.show()

def plot_image(i, predictions_array, true_label, img, class_names):

	true_label, img = true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
	                            100*np.max(predictions_array),
	                            class_names[true_label]),
	                            color=color)

def plot_value_array(i, predictions_array, true_label):

	true_label = true_label[i]
	plt.grid(False)
	plt.xticks(range(10))
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

def image_classification():

	fashion_mnist = tf.keras.datasets.fashion_mnist

	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

	y_class = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	# show_image(x_train[0])

	### Scale value in array to [0,1] ###########################

	x_train = x_train/255.0
	x_test = x_test/255.0

	### Model ###################################################

	model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28, 28)),
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.Dense(10) # output 10 value
		])

	model.compile(
			optimizer='adam',
			loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			 metrics=['accuracy']
		)

	model.fit(x_train, y_train, epochs=10)

	loss, acc = model.evaluate(x_test, y_test)

	print(f'loss:{loss} accuracy:{acc}')

	### Implement ################################################

	prob_model = tf.keras.Sequential([
				model,
				tf.keras.layers.Softmax()
			])

	predict = prob_model.predict(x_test)

	print(y_class[np.argmax(predict[0])])

	### Analysis #################################################

	num_rows = 5
	num_cols = 3

	num_images = num_rows * num_cols

	plt.figure( figsize=( 2*2*num_cols, 2*num_rows ) )

	for i in range(num_images):

		plt.subplot(num_rows, 2*num_cols, 2*i+1)
		plot_image(i, predict[i], y_test, x_test, class_names=y_class)
		plt.subplot(num_rows, 2*num_cols, 2*i+2)
		plot_value_array(i, predict[i], y_test)

	plt.tight_layout()
	plt.show()







if __name__ == '__main__':
	
	image_classification()