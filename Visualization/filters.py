import numpy as np
from keras.applications import vgg19
from keras import backend as K
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# load model
model = vgg19.VGG19(weights=None, include_top=False)
model.load_weights('/search/odin/lixudong/code_work/baidu_tiangong/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
model.summary()

# convert tensor to image
def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	x += 0.5
	x = np.clip(x, 0, 1)

	x *= 255
	x = np.clip(x, 0, 255).astype('uint8')
	return x

# Generate the filter visualization
def generate_pattern(layer_name, filter_index, size=150):
	# loss
	layer_output = model.get_layer(layer_name).output
	loss = K.mean(layer_output[:, :, :, filter_index])

	# grads
	grads = K.gradients(loss, model.input)[0]
	# Gradient normalization
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	iterate = K.function([model.input], [loss, grads])
	input_img_data = np.random.random((1, size, size, 3)) * 20 + 128

	step = 1.
	# gradient ascent
	for i in range(40):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step

	img = input_img_data[0]
	return deprocess_image(img)

# View the first 64 filters in each layer, and view only at the first layer of each convolution block
layer_name_list = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
size = 64
margin = 5

for layer_name in layer_name_list:
	print('processing filters of ' + layer_name + '...')
	results = np.zeros((8*size + 7*margin, 8*size + 7*margin, 3))
	for i in range(8):
		for j in range(8):
			filter_img = generate_pattern(layer_name, i + j * 8, size=size)

			horizontal_start = i * size + i * margin
			horizontal_end = horizontal_start + size
			vertical_start = j * size + j * margin
			vertical_end = vertical_start + size
			results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

	plt.figure(figsize=(20, 20))
	plt.title(layer_name)
	plt.imshow(results)
	plt.savefig('./filters_output/' + layer_name + '_filters.png')

