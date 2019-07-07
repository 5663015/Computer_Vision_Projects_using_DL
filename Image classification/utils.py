import tensorflow as tf


class CONFIG():
	input_shape = (150, 150, 3)
	target_size = (150, 150)
	batch_size = 128
	train_path = './data/train/'
	val_path = './data/validation/'
	test_path = './data/test/'
	epochs = 20

