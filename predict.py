from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

model = load_model("brain_mri_classifier.h5")

labels = ["No Tumor", "Tumor"]

def predict_image(img_path):
    if not os.path.exists(img_path):
        print("File not found:", img_path)
        return

    img = Image.open(img_path).convert("L")
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = img.reshape(1, 150, 150, 1)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index] * 100

    print("\n==============================")
    print("Prediction:", labels[class_index])
    print(f"Confidence: {confidence:.2f}%")
    print("==============================\n")


predict_image("brain_mri/no/no 94.JPG")
