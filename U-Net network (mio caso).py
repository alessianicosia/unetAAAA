# U-Net network

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from glob import glob
from unet_utils_mio import load_data
import cv2
from unet_utils_mio import extract_patches, preprocessing
import os
from unet_model import unet
from unet_utils_mio import datagenerator
from keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from unet_utils_mio import preprocessing


# Location of the DRIVE dataset
data_folder = 'C:/Users/aless/Desktop/Tesi/Python/unet-master/DRIVE_1/'
train_paths = glob(data_folder + 'training/images/*.jpg')
images, segmentations = load_data(train_paths)

# Print the shape of image dataset
print(images.shape)
plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
plt.title("Image DICOM")
plt.axis('off')
plt.imshow(images[0], cmap=plt.get_cmap('gray'))

plt.subplot(1, 2, 2)
plt.title("Binary Image")
plt.axis('off')
plt.imshow(segmentations[0][:, :, 0], cmap=plt.get_cmap('gray'))

plt.show()

# Divide in training and validation
train_images, val_images, train_segmentations, val_segmentations = train_test_split(
    images, segmentations, test_size=0.2, random_state=7)

# Print the shape of the training and validation datasets
print(train_images.shape)
print(train_segmentations.shape)
print(val_images.shape)
print(val_segmentations.shape)

# Work with 32x32 patches
patch_size = (32, 32)

# 200 patches per image
patches_per_im = 200

# Visualize a couple of patches as a visual check
patches, patches_segmentations = extract_patches(train_images, train_segmentations, patch_size, patches_per_im=400, seed=7)
print(patches.shape)

# Pad the validation data to fit the U-Net model
print("Old shape:", val_images.shape)
val_images, val_segmentations = preprocessing(
    val_images,
    val_segmentations,
    desired_shape=(300,300))
print("New shape:", val_images.shape)

# Use a single training image, to better demonstrate the effects of data augmentation
X_train, y_train = np.expand_dims(train_images[0], axis=0), np.expand_dims(train_segmentations[0], axis=0)
print(X_train.shape)
print(y_train.shape)

# Hyperparameters
depth = 2
channels = 32
use_batchnorm = True
batch_size = 30
epochs = 250
steps_per_epoch = int(np.ceil((patches_per_im * len(train_images)) / batch_size))

# Work with 32x32 patches
patch_size = (32, 32)

# 200 patches per image
patches_per_im = 200

# Initialize model
model = unet(input_shape=(None, None, 3), depth=depth, channels=channels, batchnorm=use_batchnorm)

# Print a summary of the model
model.summary(line_length=120)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Stop the training if the validation loss does not increase for 15 consecutive epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model with the data generator, and save the training history
history = model.fit_generator(datagenerator(X_train, y_train, patch_size, patches_per_im, batch_size),
                              validation_data=(val_images, val_segmentations),
                              steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2,
                              callbacks=[early_stopping])

# Run the model on one test image and show the results
# Test data paths
impaths_test = glob(data_folder + 'test/images/*.jpg')

# Load data
test_images, test_segmentations = load_data(impaths_test, test=True)

# Pad the data to fit the U-Net model
test_images, test_segmentations = preprocessing(test_images, test_segmentations,
                                                            desired_shape=(300, 300))

# Use a single image to evaluate
X_test = np.expand_dims(test_images[0], axis=0)

# Predict test samples
test_prediction = model.predict(X_test, batch_size=40)

# Visualize the test result
plt.figure(figsize=(12, 10))

plt.subplot(1, 3, 1)
plt.title("DICOM Image")
plt.axis('off')
plt.imshow(test_images[0], cmap=plt.get_cmap('gray'))

plt.subplot(1, 3, 2)
plt.title("Manual binary image")
plt.axis('off')
plt.imshow(test_segmentations[0][:, :, 0], cmap=plt.get_cmap('gray'))

plt.subplot(1, 3, 3)
plt.title("Predicted binary image")
plt.axis('off')
plt.imshow(test_prediction[0, :, :, 0], cmap=plt.get_cmap('gray'))

plt.show()


# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Error
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# Save U-Net
from keras.models import load_model
model.save('my_model2.h5')
del model