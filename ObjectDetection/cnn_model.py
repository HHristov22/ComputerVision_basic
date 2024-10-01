# Convolutional Neural Network with Improvements

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

def build_cnn_model(model_path="./cnn_model.keras"):
    # Part 1 - Data Preprocessing

    # dataset -> https://www.kaggle.com/datasets/pushpakhinglaspure/cats-vs-dogs/data
    dataset_path = '/home/jesus/LocalDisk/dogs_vs_cats/'

    # Preprocessing the Training set with additional augmentation
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       brightness_range=[0.8, 1.2])
    training_set = train_datagen.flow_from_directory(f'{dataset_path}/train',
                                                     target_size=(128, 128),
                                                     batch_size=32,
                                                     class_mode='binary')

    # Preprocessing the Test set
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(f'{dataset_path}/test',
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary')

    # Part 2 - Load CNN model or Build a New CNN
    if os.path.exists(model_path):
        print("Loading the existing model...")
        cnn = load_model(model_path)
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile if needed
    else:
        print("No saved model found. Building and training a new model...")

        # Initialising the CNN
        cnn = tf.keras.models.Sequential()

        # Step 1 - Convolution and Pooling
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[128, 128, 3]))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Step 2 - Additional Convolutional Layers
        cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        # Step 3 - Flattening
        cnn.add(tf.keras.layers.Flatten())

        # Step 4 - Full Connection
        cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        cnn.add(tf.keras.layers.Dropout(0.5))  # Dropout to prevent overfitting

        # Step 5 - Output Layer
        cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # Part 3 - Training the CNN
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Training the CNN on the Training set and evaluating it on the Test set
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('./best_model.keras', save_best_only=True, monitor='val_loss')

        cnn.fit(x=training_set, validation_data=test_set, epochs=25, callbacks=[early_stopping, model_checkpoint])

        # Save the trained model
        cnn.save(model_path)

    return cnn

if __name__ == '__main__':
    cnn_model = build_cnn_model()
