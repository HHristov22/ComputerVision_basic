import cnn_model_classification
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

import os
import glob

def classify_images(imgs_path):
    model = cnn_model_classification.build_cnn_model()

    for img_path in imgs_path:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            print(f"Image {img_path} -> Dog. {prediction[0][0]}")
        else:
            print(f"Image {img_path} -> Cat. {1 - prediction[0][0]}")

def list_image_files(directory):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, extension)))

    return image_files
    for file in image_files:
        print(file)


if __name__ == '__main__':
    images_path = './test_img'
    image_files = list_image_files(images_path)
    classify_images(image_files)
