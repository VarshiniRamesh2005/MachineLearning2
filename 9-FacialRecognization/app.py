# ==========================================
# Facial Recognition using ANN + Streamlit
# ORL Dataset
# ==========================================

import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================

st.set_page_config(page_title="Facial Recognition ANN", layout="wide")

st.title("Facial Recognition using Artificial Neural Network (ANN)")

st.markdown("""
### Project Description
This project implements Facial Recognition using a Multi-Layer Perceptron (ANN).
The ORL face dataset is used. Images are flattened, normalized,
reduced using PCA, and classified using ANN.
""")

# ==========================================
# LOAD DATASET FUNCTION
# ==========================================

@st.cache_resource
def load_and_train_model():

    dataset_path = "ORL"

    faces = []
    labels = []
    image_paths = []

    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)

        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (92, 112))

            faces.append(img.flatten())
            labels.append(person_folder)
            image_paths.append(image_path)

    faces = np.array(faces) / 255.0
    labels = np.array(labels)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    pca = PCA(n_components=150)
    faces_pca = pca.fit_transform(faces)

    X_train, X_test, y_train, y_test = train_test_split(
        faces_pca,
        labels_encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return {
        "model": mlp,
        "pca": pca,
        "le": le,
        "accuracy": accuracy,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "image_paths": image_paths,
        "faces": faces
    }


# ==========================================
# TRAIN MODEL (CACHED)
# ==========================================

data = load_and_train_model()

model = data["model"]
pca = data["pca"]
le = data["le"]
accuracy = data["accuracy"]
image_paths = data["image_paths"]
faces = data["faces"]

# ==========================================
# USER INPUT SECTION
# ==========================================

st.subheader("Select Image from Dataset")

selected_image_path = st.selectbox(
    "Choose an image:",
    image_paths
)

# ==========================================
# ANALYZE BUTTON
# ==========================================

if st.button("Analyze Face"):

    img = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (92, 112))

    img_flat = img.flatten().reshape(1, -1) / 255.0
    img_pca = pca.transform(img_flat)

    prediction = model.predict(img_pca)[0]

    person_name = le.inverse_transform([prediction])[0]
    person_index = prediction

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Selected Image", width=250)

    with col2:
        st.success(f"Predicted Person: {person_name}")
        st.info(f"Person Index (Encoded Label): {person_index}")

# ==========================================
# TRAINING OUTPUT BUTTON
# ==========================================

st.subheader("Model Information")

if st.button("Show Training Outputs"):

    st.write("### Model Accuracy")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    st.write("### ANN Architecture")
    st.write("Input Layer: 150 neurons (after PCA)")
    st.write("Hidden Layer 1: 256 neurons (ReLU)")
    st.write("Hidden Layer 2: 128 neurons (ReLU)")
    st.write("Output Layer: 40 neurons (Softmax)")

    st.write("### Confusion Matrix")

    cm = confusion_matrix(data["y_test"], data["y_pred"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.write("### Classification Report")
    st.text(classification_report(data["y_test"], data["y_pred"]))