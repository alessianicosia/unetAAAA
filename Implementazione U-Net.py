# Try saved U-Net
# Accuracy


from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from unet_utils_mio import load_data
from unet_utils_mio import preprocessing
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf


# Location of the DRIVE dataset
data_folder = 'C:/Users/aless/Desktop/Tesi/Python/unet-master/DRIVE_1/'
train_paths = glob(data_folder + 'training/images/*.jpg')
images, segmentations = load_data(train_paths)
model = load_model('my_model2.h5')

# Run the model on one test image and show the results
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
m = tf.keras.metrics.Accuracy()
m.update_state(test_prediction[0],test_segmentations[0])
print('Final result: ', m.result().numpy())