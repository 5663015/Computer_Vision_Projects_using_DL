import numpy as np
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

# target image and style reference image path
target_image_path = 'target1.jpg'
style_reference_image_path = 'style2.jpg'

# get size
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

# convert an image to tensor
def preprocess_image(image_path):
	img = load_img(image_path, target_size=(img_height, img_width))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img)
	return img

def deprocess_image(x):
	# The inverse operation of vgg19.preprocess_input. The aim of vgg19.preprocess_input is to 
	# subtracts the average of the pixels in ImageNet, making the center is 0
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	# convert BGR to RGB
	x = x[:, :, ::-1]

	x = np.clip(x, 0, 255).astype('uint8')
	return x

# target image
target_image = K.constant(preprocess_image(target_image_path))
# style image
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
# Generated image
combination_image = K.placeholder((1, img_height, img_width, 3))
# Merge [target_image, style_reference_image, combination_image] into a batch
input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)

# load model
model = vgg19.VGG19(input_tensor=input_tensor, weights=None, include_top=False)
model.load_weights('/search/odin/lixudong/code_work/baidu_tiangong/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
model.summary()

# content loss
def content_loss(base, combination):
	return K.sum(K.square(combination - base))

# gram matrix
def gram_matrix(x):
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram

# style loss
def style_loss(style, combination):
	S = gram_matrix(style)
	C = gram_matrix(combination)
	size = img_height * img_width
	channels = 3
	return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# total variation loss, make the generated image have spatial continuity
def total_variation_loss(x):
	a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
	b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))

# dict of all layers
output_dict = dict([(layer.name, layer.output) for layer in model.layers])
# Layer for content loss
content_layer = 'block5_conv2'
# Layers for style loss
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Weight of three losses
total_variation_weight = 1e-4
style_weight = 0.8
content_weight = 0.025

loss = K.variable(0.)
# add content loss 
layer_features = output_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)
# add style loss
for layer_name in style_layers:
	layer_features = output_dict[layer_name]
	style_reference_features = layer_features[1, :, :, :]
	combination_features = layer_features[2, :, :, :]
	sl = style_loss(style_reference_features, combination_features)
	loss += (style_weight / len(style_layers)) * sl
# add total variation loss
loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

# Get loss and grads, used in scipy.optimize
class Evaluator(object):
	def __init__(self):
		self.loss_value = None
		self.grad_values = None

	def loss(self, x):
		assert self.loss_value is None
		x = x.reshape((1, img_height, img_width, 3))
		outs = fetch_loss_and_grads([x])
		loss_value = outs[0]
		grad_values = outs[1].flatten().astype('float64')
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values

evaluator = Evaluator()

result_prefix = 'target1_style2'
iterations = 15

x = preprocess_image(target_image_path)
x = x.flatten()		# flatten x, scipy.optimize.fmin_l_bfgs_b only process flattened vectors
time1 = time.clock()
for i in range(iterations):
	print('Start of iteration: ', i, '==================')
	x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxiter=50, maxfun=50)
	print('current loss value:', min_val)
	img = x.copy().reshape((img_height, img_width, 3))
	img = deprocess_image(img)
	fname = result_prefix + '_at_iteration%d.png' % i
	imsave('./output/' + fname, img)
time2 = time.clock()
print('running time:', str((time2 - time1)/60))

