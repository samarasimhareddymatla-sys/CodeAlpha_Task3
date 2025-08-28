import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
model = load_model("handwritten_model.h5")
st.title("🖊 Handwritten Digit Recognition")
st.write("Upload a digit image (28x28) and the model will predict it.")
uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE
    img_resized = cv2.resize(img, (28,28))
    st.image(img_resized, caption="Uploaded Digit", width=150)
    img_resized = img_resized.reshape(1, 28, 28, 1).astype("float32") / 2
    prediction = np.argmax(model.predict(img_resized), axis=1)
    st.success(f"Predicted Digit: {prediction[0]}")
