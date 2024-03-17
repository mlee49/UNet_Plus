import os
import io
import random
import nibabel
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.config.experimental.list_physical_devices('GPU'):
    print("TensorFlow will run on GPU")
else:
    print("TensorFlow will run on CPU")

image_directory = 'MRI/test_anatomical/' #edit
mask_directory = 'MRI/test_liver_seg/' #edit
save_dir = 'MRI/run_save_1' #edit

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []

SIZE = 224

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

checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=5, 
                    validation_data=(X_test, y_test), 
                    shuffle=False,
                    callbacks = [checkpoint])

model.save('liver.keras')


#chart


model = get_model()
model.load_weights('best_model.h5')  # Ensure this matches the filename used in the checkpoint


def save_combined_image_with_labels(image, true_mask, predicted_mask, index, save_dir):
    # Set up the figure
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    
    # Display the anatomical image
    axs[0].imshow(image, cmap='gray')
    axs[0].title.set_text('Anatomical Image')
    axs[0].axis('off')

    # Display the true mask
    axs[1].imshow(true_mask, cmap='gray')
    axs[1].title.set_text('True Mask')
    axs[1].axis('off')

    # Display the predicted mask
    axs[2].imshow(predicted_mask, cmap='gray')
    axs[2].title.set_text('Predicted Mask')
    axs[2].axis('off')

    # Create and display the overlay image
    red_cmap = ListedColormap(['none','red'])  # Create a custom colormap for the predicted mask
    axs[3].imshow(true_mask, cmap='gray')  # Overlay true mask with blue color and some transparency
    axs[3].imshow(predicted_mask, cmap=red_cmap, alpha = 0.5)  # Overlay predicted mask with red color and some transparency
    axs[3].title.set_text('Overlay Mask')
    axs[3].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'combined_labeled_{index}.png'))
    plt.close(fig)

# Iterate through the test set and save the combined images with labels
for i in range(len(X_test)):
    test_img = X_test[i]
    ground_truth = y_test[i]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

    # Save the combined image with labels
    save_combined_image_with_labels(test_img[:,:,0], ground_truth[:,:,0], prediction, i, save_dir)


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


# # Select random test images
# test_img_number = random.randint(0, len(X_test) - 1)
# test_img = X_test[test_img_number]
# ground_truth = y_test[test_img_number]
# test_img_input = np.expand_dims(test_img, 0)
# prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

# test_img_number2 = random.randint(0, len(X_test) - 1)
# test_img2 = X_test[test_img_number2]
# ground_truth2 = y_test[test_img_number2]
# test_img_input2 = np.expand_dims(test_img2, 0)
# prediction2 = (model.predict(test_img_input2)[0,:,:,0] > 0.5).astype(np.uint8)

# # Visualize the results
# plt.figure(figsize=(16, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='gray')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(prediction, cmap='gray')
# plt.subplot(234)
# plt.title('Testing Image')
# plt.imshow(test_img2[:,:,0], cmap='gray')
# plt.subplot(235)
# plt.title('Testing Label')
# plt.imshow(ground_truth2[:,:,0], cmap='gray')
# plt.subplot(236)
# plt.title("Prediction on test image")
# plt.imshow(prediction2, cmap='gray')
# plt.show()
