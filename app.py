import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# -----------------------------
# Load ONNX model
# -----------------------------
onnx_model = "saved_model.onnx"
session = ort.InferenceSession(onnx_model)

# Class labels (make sure they match your dataset classes order)
class_labels = ["Bad", "Good"]

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch
    return img

# -----------------------------
# OpenCV check for blur
# -----------------------------
def check_blur(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return "Bad Image (Blur)" if laplacian_var < 100 else "Good Image"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🖼️ AI Image Quality Analyzer")
st.write("Upload an image to check if it's **Good** or **Bad** (Blur, Low contrast, etc).")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # OpenCV analysis
    opencv_result = check_blur(image)

    # CNN prediction
    img_input = preprocess_image(image)
    inputs = {session.get_inputs()[0].name: img_input}
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0])
    cnn_result = class_labels[pred]

    # Show results
    st.subheader("Results:")
    st.write(f"**OpenCV Analysis:** {opencv_result}")
    st.write(f"**CNN Model Prediction:** {cnn_result}")
