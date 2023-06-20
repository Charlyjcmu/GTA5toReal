# Download the dataset

from google.colab import drive
drive.mount('/content/gdrive')

# Initialize the model

from keras_segmentation.models.unet import vgg_unet
import tensorflow as tf
import keras_segmentation

new_model = tf.keras.models.load_model('gdrive/MyDrive/10-417/vgg_unet_padded_full')
new_model.save_weights('/tmp/weights')

model = vgg_unet(n_classes=20,  input_height=1056, input_width=2048)
model.load_weights('/tmp/weights')

from IPython.display import Image

image_path = "gdrive/MyDrive/10-417/frankfurt_000001_047552.png"

# no overlay
o = model.predict_segmentation(
    inp=image_path,
    out_fname="/tmp/out.png" , overlay_img=False, show_legends=True,
    class_names = ["Unlabeled", "Road",    "Pavement", "Building","Bicyclist","SignSymbol","Car","Pole", "Fence", "Tree","Vegetation", "Sky"]
)

Image('/tmp/out.png')

from IPython.display import Image

# with overlay
o = model.predict_segmentation(
    inp=image_path,
    out_fname="/tmp/out.png" , overlay_img=True, show_legends=True,
    class_names = ["Unlabeled", "Road",    "Pavement", "Building","Bicyclist","SignSymbol","Car","Pole", "Fence", "Tree","Vegetation", "Sky"]
)

Image('/tmp/out.png')