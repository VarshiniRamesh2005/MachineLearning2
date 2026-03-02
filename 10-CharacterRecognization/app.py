# ==========================================================
# Handwritten Character Recognition (A–Z) using MLP
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(page_title="Character Recognition using MLP", layout="wide")

st.title("🔤 Handwritten Character Recognition (A–Z)")
st.markdown("""
This project recognizes handwritten English alphabets (A–Z)
using a Multi-Layer Perceptron (MLP) classifier trained on pixel data.
Upload a 28x28 grayscale character image for prediction.
""")

st.markdown("---")

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():
    data = pd.read_csv("A_Z Handwritten Data.csv", nrows=50000).astype("float32")
    return data

with st.spinner("Loading dataset..."):
    data = load_data()

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

letters = [chr(i) for i in range(65, 91)]
y_letters = np.array([letters[int(label)] for label in y])

# ==========================================================
# TRAIN MODEL (ONLY FIRST TIME)
# ==========================================================

model_exists = os.path.exists("model.pkl")

if not model_exists:

    st.info("Training model for first time...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_letters, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=42
    )

    with st.spinner("Training MLP model..."):
        mlp.fit(X_train, y_train)

    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average='weighted', zero_division=0
    )

    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='weighted', zero_division=0
    )

    test_cm = confusion_matrix(y_test, y_test_pred, labels=letters)

    metrics = {
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': test_cm
    }

    joblib.dump((mlp, scaler, metrics), "model.pkl")
    st.success("Model trained and saved successfully!")

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model, scaler, metrics = joblib.load("model.pkl")

# ==========================================================
# IMAGE UPLOAD SECTION
# ==========================================================

st.subheader("📤 Upload Character Image")

uploaded_file = st.file_uploader(
    "Browse and upload a handwritten character image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image)

    # Normalize display properly
    display_image = image_array.astype(np.uint8)

    # Flatten for prediction
    image_flat = image_array.reshape(1, -1)
    image_scaled = scaler.transform(image_flat)

    if st.button("Analyze Character"):

        prediction = model.predict(image_scaled)

        col1, col2 = st.columns(2)

        with col1:
            st.image(display_image, caption="Uploaded Image", width=200)

        with col2:
            st.success(f"### Predicted Character: {prediction[0]}")

st.markdown("---")

# ==========================================================
# SHOW TRAINING OUTPUTS BUTTON
# ==========================================================

if st.button("📊 Show Training Outputs"):

    st.subheader("Training Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Training Data")
        st.write(f"Accuracy: {metrics['train_accuracy']*100:.2f}%")
        st.write(f"Precision: {metrics['train_precision']*100:.2f}%")
        st.write(f"Recall: {metrics['train_recall']*100:.2f}%")
        st.write(f"F1-score: {metrics['train_f1']*100:.2f}%")

    with col2:
        st.markdown("### Testing Data")
        st.write(f"Accuracy: {metrics['test_accuracy']*100:.2f}%")
        st.write(f"Precision: {metrics['test_precision']*100:.2f}%")
        st.write(f"Recall: {metrics['test_recall']*100:.2f}%")
        st.write(f"F1-score: {metrics['test_f1']*100:.2f}%")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        metrics['confusion_matrix'],
        cmap='Blues',
        xticklabels=letters,
        yticklabels=letters,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
    plt.close()