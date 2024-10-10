import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only errors and warnnings

import numpy as np
import tensorflow as tf

# Download and prepare the data
from scripts.data_preparation import download_dataset, prepare_data
data_dir = './data'
download_dataset(data_dir)
prepare_data(data_dir)

images_path = rf"./test_data"

# Check model
model_filename = 'model_128.keras'
if os.path.exists(model_filename):
    print(f"Loading model: {model_filename}...")
    model = tf.keras.models.load_model(model_filename)
else:
    print("Creat new model...")
    from scripts.train import DataGenerator, unet_model
    from scripts.utils import list_files
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.model_selection import train_test_split

    # images_path = os.path.join(data_dir, 'images')
    masks_path = os.path.join(data_dir, 'annotations', 'trimaps')

    image_files = list_files(images_path, ['.jpg'])
    mask_files = list_files(masks_path, ['.png'])

    image_files.sort()
    mask_files.sort()

    train_img_files, val_img_files, train_mask_files, val_mask_files = train_test_split(
        image_files, mask_files, test_size=0.1, random_state=42
    )

    batch_size = 4
    image_size = (128, 128)

    train_generator = DataGenerator(train_img_files, train_mask_files, batch_size, image_size)
    val_generator = DataGenerator(val_img_files, val_mask_files, batch_size, image_size)

    model = unet_model(input_size=(image_size[0], image_size[1], 3))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_filename, save_best_only=True)
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[checkpoint, early_stopping]
    )

    model.save(model_filename)
    print(f"Model saved: {model_filename}.")

# Predict
from scripts.predict import load_test_data
X_test, image_files = load_test_data(images_path, image_size=(128, 128))

preds = model.predict(X_test)
preds = np.argmax(preds, axis=-1)

import matplotlib.pyplot as plt

for i in range(len(X_test)):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(X_test[i])
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(preds[i], cmap='gray')
    plt.title('Predict mask')
    plt.axis('off')

    plt.show()
