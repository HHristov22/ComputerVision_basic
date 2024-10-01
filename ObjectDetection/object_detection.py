import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from cnn_model import build_cnn_model

def preprocess_roi(roi, target_size=(128, 128)):
    roi_resized = cv2.resize(roi, target_size)
    roi_array = img_to_array(roi_resized)
    roi_array = np.expand_dims(roi_array, axis=0)
    roi_array = roi_array / 255.0
    return roi_array

def classify_roi(model, roi_array, class_names):
    prediction = model.predict(roi_array)
    if prediction[0][0] > 0.5:
        label = class_names[1]  # 'dog'
        confidence = prediction[0][0]
    else:
        label = class_names[0]  # 'cat'
        confidence = 1 - prediction[0][0]
    return label, confidence

def resize_image(image, target_height=480):
    (h, w) = image.shape[:2]
    ratio = target_height / float(h)
    dim = (int(w * ratio), target_height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

def detect_and_classify(image_path, model, class_names, min_contour_area=100, target_size=(128, 128)):

    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    image = resize_image(image, target_height=480)
    output_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(thresh, threshold1=30, threshold2=100)

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            roi_array = preprocess_roi(roi, target_size)
            label, confidence = classify_roi(model, roi_array, class_names)

            if confidence > 0.5:
                label_text = f"{label}: {confidence*100:.2f}%"
            else:
                label_text = "Unknown"

            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(6, 6))
    # plt.imshow(output_rgb)
    # plt.axis('off')
    # plt.show()

    cv2.imwrite('test_result.jpg', output_image)

if __name__ == "__main__":
    model_path = './cnn_model.keras'
    image_path = './test1.jpg'
    image_path = './test2.jpg'
    image_path = './test3.jpeg'

    model = build_cnn_model(model_path)

    print("************ CNN MODEL ***********")

    class_names = ['cat', 'dog']

    detect_and_classify(image_path, model, class_names, min_contour_area=150, target_size=(128, 128))
