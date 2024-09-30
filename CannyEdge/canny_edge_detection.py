import cv2
import numpy as np
import os


def resize_image(img, new_height=480):
    original_height, original_width = img.shape

    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def canny_edge_detection(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    img = resize_image(img)
    print(img.shape)

    # Reduce noise
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred_img, 80, 180, L2gradient=True)
    # edges = cv2.Canny(blurred_img, 80, 190)

    # Save the edge-detected image
    file_name, file_extension = os.path.splitext(image_path)
    new_file_name = f"{file_name}_edge{file_extension}"
    cv2.imwrite(new_file_name, edges)


if __name__ == "__main__":
    images = [rf"./test1.jpg", rf"./test2.png"]
    for image_path in images:
        canny_edge_detection(image_path)
