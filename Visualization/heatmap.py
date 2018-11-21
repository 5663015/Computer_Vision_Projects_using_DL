import numpy as np
import cv2
from keras.applications import vgg19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# load model
model = vgg19.VGG19(weights=None, include_top=True)
model.load_weights('/search/odin/lixudong/computer_vision_projects/vgg19_weights_tf_dim_ordering_tf_kernels.h5')
model.summary()

# load image
img_path = 'duckbill.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

# prediction result
preds = model.predict(x)
print('predicted:', decode_predictions(preds, top=3)[0])
# get the index
index = np.argmax(preds[0])

# The platypus element in the prediction vector
output = model.output[:, index]
# the last conv layer of vgg19
last_conv_layer = model.get_layer('block5_conv4')

# Grad-CAM algorithm
grads = K.gradients(output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
	conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

# Heatmap postprocessing
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# save heatmap
plt.figure(figsize=(20, 20))
plt.title('heatmap')
plt.matshow(heatmap)
plt.savefig('./heatmap_output/heatmap.png')

# Stack the heatmap with the original one
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./heatmap_output/duckbill_cam.jpg', superimposed_img)

