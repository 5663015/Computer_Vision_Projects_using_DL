import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import vgg19
from keras.models import Model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# target image size
image_size = 200

# load model
model = vgg19.VGG19(weights=None, input_shape=(image_size, image_size, 3), include_top=False)
model.load_weights('/search/odin/lixudong/code_work/baidu_tiangong/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
model.summary()

# image path
img_path = 'cat.jpg'
# load and proccess image
img = image.load_img(img_path, target_size=(image_size, image_size))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
img_tensor = vgg19.preprocess_input(img_tensor)

# Instantiated model with a input tensor and a list of output tensors
layer_outputs = [layer.output for layer in model.layers[1:-1]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

# names of layers
layer_names = []
for layer in model.layers[1:-1]:
	layer_names.append(layer.name)

images_per_row = 16

# show feature maps
for layer_name, layer_activation in zip(layer_names, activations):
	# shape of feature maps: (1, size, size, n_features)
	# the number of feature maps
	n_features = layer_activation.shape[-1]
	# size of feature maps
	size = layer_activation.shape[1]

	n_cols = n_features // images_per_row
	display_grid = np.zeros((size*n_cols, images_per_row*size))

	for col in range(n_cols):
		for row in range(images_per_row):
			channel_image = layer_activation[0, :, :, col*images_per_row + row]
			channel_image -= channel_image.mean()
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col*size:(col+1)*size, row*size:(row+1)*size] = channel_image

	plt.figure(figsize=(display_grid.shape[1]/size, display_grid.shape[0]/size))
	plt.title(layer_name)
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig('./output/' + layer_name + 'activation.png')