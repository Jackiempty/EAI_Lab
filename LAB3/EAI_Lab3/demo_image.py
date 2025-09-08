import tensorflow as tf
import click
import cv2
import numpy as np
import importlib
import time

from estimation.scripts.config import get_default_configuration
from estimation.scripts.coordinates import get_coordinates
from estimation.scripts.connections import get_connections
from estimation.scripts.estimators import estimate
from estimation.scripts.renderers import draw

from tf_netbuilder_ext.extensions import register_tf_netbuilder_extensions

# ↓ Fill your own ↓ #
model_weights_path = "./model_weights/openpose"
input_image = "./images/1.jpg"
output_image = "output.png"
# ↑ Fill your own ↑ #

register_tf_netbuilder_extensions()

module = importlib.import_module('model')
create_model = getattr(module, "create_openpose")
model = create_model()
model.load_weights(model_weights_path) 

img = cv2.imread(input_image)  # B,G,R order
input_img = img[np.newaxis, :, :, [2, 1, 0]]
inputs = tf.convert_to_tensor(input_img)

outputs = model.predict(inputs)
pafs = outputs[10][0, ...]
heatmaps = outputs[11][0, ...]
cfg = get_default_configuration()
coordinates = get_coordinates(cfg, heatmaps)
connections = get_connections(cfg, coordinates, pafs)

# ↓ You can analsis the time of your algo. ↓ #
x = time.time()
skeletons = estimate(cfg, connections)
y = time.time()
print(y-x) # Inference time
# ↑ You can analsis the time of your algo. ↑ #

output = draw(cfg, img, coordinates, skeletons, resize_fac=8)

cv2.imwrite(output_image, output)

print(f"Output saved: {output_image}")