import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from .model import unet_model
from .utils import list_files
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, image_files, mask_files, batch_size=4, image_size=(128, 128)):
        self.image_files = image_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []

        for img_path, mask_path in zip(batch_x, batch_y):
            img = load_img(img_path, target_size=self.image_size)
            img = img_to_array(img) / 255.0
            images.append(img)

            mask = load_img(mask_path, target_size=self.image_size, color_mode='grayscale')
            mask = img_to_array(mask)
            mask = mask[..., 0] - 1  # The values in the masks are 1,2,3 - subtract 1 to make 0,1,2
            masks.append(mask)

        X = np.array(images)
        y = np.array(masks)
        y = np.expand_dims(y, axis=-1)

        return X, y

if __name__ == "__main__":
    data_dir = './data'
    images_path = os.path.join(data_dir, 'images')
    masks_path = os.path.join(data_dir, 'annotations', 'trimaps')

    # List of images
    image_files = list_files(images_path, ['.jpg'])
    mask_files = list_files(masks_path, ['.png'])

    image_files.sort()
    mask_files.sort()
    from sklearn.model_selection import train_test_split
    train_img_files, val_img_files, train_mask_files, val_mask_files = train_test_split(
        image_files, mask_files, test_size=0.1, random_state=42
    )

    batch_size = 4
    image_size = (128, 128)

    train_generator = DataGenerator(train_img_files, train_mask_files, batch_size, image_size)
    val_generator = DataGenerator(val_img_files, val_mask_files, batch_size, image_size)

    model = unet_model(input_size=(image_size[0], image_size[1], 3))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('./model_128.keras', save_best_only=True)
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[checkpoint, early_stopping]
    )
