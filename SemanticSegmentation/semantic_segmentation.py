import cnn_model
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import os
import glob

def segmentation_images(imgs_path):
    model = cnn_model.build_cnn_model()

    # Identify the last convolutional layer's name in your model
    # Replace 'conv2d_3' with the actual layer name from your model
    last_conv_layer_name = 'conv2d_3'

    for img_path in imgs_path:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array_expanded = np.expand_dims(img_array, axis=0)

        # Use Grad-CAM to get the heatmap
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array_expanded)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        guided_grads = grads[0]
        conv_outputs = conv_outputs[0]

        # Compute the guided gradients
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * conv_outputs[:, :, i]

        cam = cv2.resize(cam.numpy(), (128, 128))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        # Overlay the heatmap on the image
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        img_original = cv2.cvtColor(np.uint8(img_array * 255), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

        # Save the image with segmentation
        output_dir = './output_segmentation'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, overlay)
        print(f"Segmentation image saved at {output_path}")

def list_image_files(directory):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, extension)))

    return image_files

if __name__ == '__main__':
    images_path = './test_img'  # Replace with your image directory
    image_files = list_image_files(images_path)
    segmentation_images(image_files)
