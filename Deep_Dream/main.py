import numpy as np
import scipy
import keras.backend as K
from keras.applications import inception_v3 
from keras.preprocessing import image

# We don't train models, so this code will forbid operations related to training
K.set_learning_phase(0)

# load model
model = inception_v3.InceptionV3(weights=None, include_top=False)
model.load_weights('/search/odin/lixudong/code_work/baidu_tiangong/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
model.summary()

# DeepDream setting
# Map the name of the layer to a coefficient
layer_contributions = {
	'mixed1': 2.,
	'mixed2': 3., 
	'mixed3': 5.,
	'mixed4': 2, 
	'mixed5': 1.0, 
	'mixed6': 0.5,
	'mixed7': 0.5,
	'mixed8': 0.5,
	'mixed9': 0.5,
}

# maximize loss
layer_dict = dict([(layer.name, layer) for layer in model.layers])
loss = K.variable(0.)
for layer_name in layer_contributions:
	coeff = layer_contributions[layer_name]
	activation = layer_dict[layer_name].output
	scaling = K.prod(K.cast(K.shape(activation), 'float32'))
	loss += coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling

dream = model.input 	# input image
grads = K.gradients(loss, dream)[0]		# Calculate the gradient of the loss relative to the dream image
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)		# Normalize the gradient

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

# get loss and grads
def eval_loss_and_grads(x):
	outs = fetch_loss_and_grads([x])
	loss_value = outs[0]
	grad_values = outs[1]
	return loss_value, grad_values

# gradient ascent
def gradient_ascent(x, iterations, step, max_loss=None):
	for i in range(iterations):
		loss_value, grad_values = eval_loss_and_grads(x)
		if max_loss is not None and loss_value > max_loss:
			break
		print('Loss value at ', i, ':', loss_value)
		x += step * grad_values
	return x

# resize the image
def resize_img(img, size):
	img = np.copy(img)
	factors = (1, float(size[0])/img.shape[1], float(size[1])/img.shape[2], 1)
	return scipy.ndimage.zoom(img, factors, order=1)

# save image
def save_img(img, fname):
	pil_img = deprocess_image(np.copy(img))
	scipy.misc.imsave(fname, pil_img)

# open iamge, change shape of iamge and convert the image format to a tensor that Inception can handle 
def preprocess_image(image_path):
	img = image.load_img(image_path)
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = inception_v3.preprocess_input(img)
	return img

# convert tensor to image
def deprocess_image(x):
	x = x.reshape((x.shape[1], x.shape[2], 3))
	x /= 2.
	x += 0.5
	x *= 255
	x = np.clip(x, 0, 255).astype('uint8')
	return x

step = 0.01			# step of gradient ascent
num_octave = 3 		# number of octave
octave_scale = 1.4 	# The scale between the two octave
iterations = 100 	# iterations of running gradient ascent in each octave

max_loss = 20. 	
# list of image names (Suffixes are not included)
base_image_name_list = ['base_image1', 'base_image2', 'base_image3', 'base_image4', 'base_image5']

# create dream images
def create_dream(base_image_name):
	# load image
	img = preprocess_image('./' + base_image_name + '.jpg')
	original_shape = img.shape[1:3]

	# Define different octave of run gradient rise
	sucessive_shapes = [original_shape]
	for i in range(1, num_octave):
		shape = tuple([int(dim/(octave_scale ** i)) for dim in original_shape])
		sucessive_shapes.append(shape)

	# Flip the list into ascending order
	sucessive_shapes = sucessive_shapes[::-1]

	original_img = np.copy(img)
	shrunk_original_img = resize_img(img, sucessive_shapes[0])

	for shape in sucessive_shapes:
		print('Processing image shape', shape)
		img = resize_img(img, shape)
		img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
		upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
		same_size_original = resize_img(original_img, shape)
		# The difference between the same_size_original and upscaled_shrunk_original_img is the lost details in the resize process
		lost_detail = same_size_original - upscaled_shrunk_original_img

		img += lost_detail
		shrunk_original_img = resize_img(original_img, shape)
	save_img(img, fname='./output/' + base_image_name + '_final_dream.png')

for base_image_name in base_image_name_list:
	print('process ' + base_image_name + '...')
	create_dream(base_image_name)