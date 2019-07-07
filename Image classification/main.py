import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn import metrics
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3, 4, 5, 6, 7'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils import *

config_ = tf.ConfigProto()  
config_.gpu_options.per_process_gpu_memory_fraction = 0.9  
config_.gpu_options.allow_growth = True   
sess = tf.Session(config = config_)
with tf.Session(config=config_) as sess:
	pass

config = CONFIG()

# create model
def create_model():
	# use pre-trained model
	base_model = tf.keras.applications.InceptionV3(include_top=False, 
		weights='imagenet', input_shape=config.input_shape)
	x = base_model.output
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(512, activation='relu')(x)
	output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	model = tf.keras.Model(inputs=base_model.input, outputs=output)
	for i in range(280):
		model.layers[i].trainable = False
	# multi gpus model
	parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=6)
	# compile model
	parallel_model.compile(loss='binary_crossentropy', 
		optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['acc'])
	return parallel_model, model

def train(train_path, val_path, model, original_model):
	# train data generator
	train_datagen = ImageDataGenerator(rescale=1./255)
	train_generator = train_datagen.flow_from_directory(
		train_path, 
		target_size=config.target_size,
		batch_size=config.batch_size,
		class_mode='binary'
		)
	# test data generator
	val_datagen = ImageDataGenerator(rescale=1./255)
	val_generator = val_datagen.flow_from_directory(
		val_path, 
		target_size=config.target_size,
		batch_size=config.batch_size,
		class_mode='binary'
		)
	# train
	history = model.fit_generator(train_generator, epochs=config.epochs, 
		validation_data=val_generator)
	original_model.save('./models/model.h5')
	print(history.history)
	plot(history.history)

def test(test_path):
	# load model
	model = tf.keras.models.load_model('./models/model.h5')
	model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['acc'])
	# test data generator
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(
		test_path, 
		target_size=config.target_size,
		batch_size=1,
		class_mode='binary'
		)
	# predict
	prediction, labels = [], []
	i = 0
	for batch_x, batch_y in test_generator:
		if i == 2000:
			break
		pred = model.predict(batch_x)
		prediction.append(pred)
		labels.append(batch_y)
		i += 1
	prediction, labels = np.array(prediction).reshape((-1, 1)), np.array(labels).reshape((-1, 1))
	prediction = np.where(prediction >= 0.5, 1, prediction)
	prediction = np.where(prediction < 0.5, 0, prediction)

	# 评估
	print("Precision, Recall and F1-Score...")
	print(metrics.classification_report(labels, prediction, target_names=['cat', 'dog']))
	# 混淆矩阵
	print("Confusion Matrix...")
	cm = metrics.confusion_matrix(labels, prediction)
	print(cm)
'''
# create model
model, original_model = create_model()
# train model
train(config.train_path, config.val_path, model, original_model)
'''
# test model
test(config.test_path)


