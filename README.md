Brain Tumor Detection using CNN
This project is a deep learning–based classification system that detects the presence of a brain tumor from MRI scan images using a Convolutional Neural Network (CNN). The model was trained on a labeled dataset containing images with Tumor and No Tumor cases. The goal of this project is to demonstrate the use of computer vision and deep learning for medical imaging applications.
The repository includes the complete workflow—data preprocessing, model training, evaluation, and prediction script—to classify MRI images with high accuracy. After training, the model is saved as brain_mri_classifier.h5, and you can test any MRI scan using the provided predict.py script.

Features:
CNN-based binary image classification
MRI brain scan analysis
Trained using TensorFlow and Keras
Prediction script that displays classification and confidence
Easy-to-run project structure

Repository Structure:
|-- brain_mri_classifier.h5
|-- train.py
|-- predict.py
|-- dataset/
      |-- yes/
      |-- no/
|-- README.md

Dataset Used:
The dataset contains labeled MRI scans categorized into:
yes: tumor present
no: tumor absent

Results:
The model outputs:
Predicted class: Tumor / No Tumor
Confidence score percentage
