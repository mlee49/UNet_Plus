import os
import io
import random
import nibabel
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import Sequence
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from simple_unet_plus_model import simple_unet_plus_model
from skimage.transform import resize

image_directory = 'MRI Images/Test_Anatomical_Images_1/'
mask_directory = 'MRI Images/Test_Liver_Images_1/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []

SIZE = 128

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        image = resize(image, (SIZE, SIZE))
        image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(mask_directory+image_name)
        image = np.array(image.get_fdata())
        image = resize(image, (SIZE, SIZE))
        mask_dataset.append(np.array(image))

for i in range(len(image_dataset)):
    for j in range(image_dataset[i].shape[2]):
        sliced_image_dataset.append(image_dataset[i][:,:,j])

for i in range(len(mask_dataset)):
    for j in range(mask_dataset[i].shape[2]):
        if i == 16 and j == 25:
            continue
        else:
            sliced_mask_dataset.append(mask_dataset[i][:,:,j])

#Normalize images
sliced_image_dataset = np.expand_dims(np.array(sliced_image_dataset),3)
#D not normalize masks, just rescale to 0 to 1.
sliced_mask_dataset = np.expand_dims((np.array(sliced_mask_dataset)),3)

X_train, X_test, y_train, y_test = train_test_split(sliced_image_dataset, sliced_mask_dataset, test_size = 0.20, random_state = 0)

##############################################################################

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model():
    return simple_unet_plus_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=10, 
                    validation_data=(X_test, y_test), 
                    shuffle=False,
                    callbacks = [checkpoint])

model.save('liver.keras')


#chart

plt.figure(figsize=(12, 6))

# Plot for loss
plt.subplot(121)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot for Dice coefficient - ensure your model's metrics include 'dice_coef'
plt.subplot(122)
plt.plot(history.history['dice_coef'], label='Training Dice Coefficient')  # Adjust key if necessary
plt.plot(history.history['val_dice_coef'], label='Validation Dice Coefficient')  # Adjust key if necessary
plt.title('Dice Coefficient over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

model.save('liver_segmentation_model.keras')

model = get_model()
model.load_weights('best_model.h5')  # Ensure this matches the filename used in the checkpoint

# Select random test images
test_img_number = random.randint(0, len(X_test) - 1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

test_img_number2 = random.randint(0, len(X_test) - 1)
test_img2 = X_test[test_img_number2]
ground_truth2 = y_test[test_img_number2]
test_img_input2 = np.expand_dims(test_img2, 0)
prediction2 = (model.predict(test_img_input2)[0,:,:,0] > 0.5).astype(np.uint8)

# Visualize the results
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('Testing Image')
plt.imshow(test_img2[:,:,0], cmap='gray')
plt.subplot(235)
plt.title('Testing Label')
plt.imshow(ground_truth2[:,:,0], cmap='gray')
plt.subplot(236)
plt.title("Prediction on test image")
plt.imshow(prediction2, cmap='gray')
plt.show()