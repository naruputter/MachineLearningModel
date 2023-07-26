import tensorflow as tf

def benginning():

	mnist = tf.keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train/255.0, x_test/255.0

	model = tf.keras.models.Sequential([
			tf.keras.layers.Flatten(input_shape=(28, 28)),
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(10)
		])

	### Test Pretrain ##################################


	predictions = model(x_train[:1]).numpy()
	print(predictions)
	print(tf.nn.softmax(predictions).numpy()) # softmax will change predict value to probability

	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	print(loss_fn(y_train[:1], predictions).numpy())

	### train #######################################

	model.compile(

		optimizer='adam',
		loss=loss_fn,
		metrics=['accuracy']
	)

	model.fit(x_train, y_train, epochs=5)

	### test #############################################

	print(model.evaluate(x_test, y_test, verbose=2))

	probability_model = tf.keras.Sequential([
			model,
			tf.keras.layers.Softmax()
		])

	print(probability_model(x_test[:5]))


if __name__ == '__main__':
	
	benginning()