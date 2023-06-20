from keras_segmentation.models.unet import vgg_unet
from IPython.display import Image

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

# load the pre-trained model with weights trained from full GTA training set
model = tf.keras.models.load_model('vgg_unet_padded')
model.save_weights('temp_weights/weights')
model = vgg_unet(n_classes=20,  input_height=320, input_width=640)
model.load_weights('temp_weights/weights')

# train the model on the augmented data
model.train(
    train_images =  "aug1/images_prepped_train/",
    train_annotations = "aug1/annotations_prepped_train/",
    checkpoints_path = "padded/resnet_unet_3" , epochs=5,
    validate=True,
    val_images = "padded/images_prepped_test", 
    val_annotations = "padded/annotations_prepped_test"
)

print(model.history.history.keys())
acc = model.history.history['val_accuracy']
print(acc)

train_acc = model.history.history['accuracy']
print(train_acc)

# plot the training and validation accuracy results
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# obtain test results for the model
print(model.evaluate_segmentation(inp_images_dir="padded/cityscapes-images_prepped_test/"  , annotations_dir="padded/cityscapes-annotations_prepped_test/"))

# save the weights of the models
model.save("vgg_unet_aug1_all_data")