import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from .utils import list_files
# from utils import list_files
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def load_test_data(images_path, image_size=(128, 128), num_samples=10):
    image_files = list_files(images_path, ['.jpg', '.jpeg', '.png'])
    image_files.sort()
    image_files = image_files[:num_samples]

    X = []
    for img_path in image_files:
        img = load_img(img_path, target_size=image_size)
        img = img_to_array(img) / 255.0
        X.append(img)

    X = np.array(X)
    return X, image_files

if __name__ == "__main__":
    data_dir = './data'
    test_images_path = os.path.join(data_dir, 'images')
    test_images_path = rf"./test_data"

    X_test, image_files = load_test_data(test_images_path)
    print(X_test, image_files)
    model = load_model('./model_128.keras')

    preds = model.predict(X_test)
    preds = np.argmax(preds, axis=-1)

    for i in range(len(X_test)):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(X_test[i])
        plt.title('Original image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(preds[i], cmap='gray')
        plt.title('Predict image')
        plt.axis('off')

        plt.show()
