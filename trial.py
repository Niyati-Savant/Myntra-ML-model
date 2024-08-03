import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import keras
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from scipy.spatial.distance import cosine

feature_list = np.array(pickle.load(open('embeddings_new.pkl', 'rb')))
filenames = pickle.load(open('filenames_new.pkl', 'rb'))

model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = keras.Sequential([
    model,
    keras.layers.GlobalMaxPooling2D()
])

st.title('Myntra Insta-Buy')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = keras.applications.resnet.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return distances, indices


# Calculate similarity percentage
def calculate_similarity_percentage(features, feature_list, indices):
    similarities = []
    for index in indices[0]:
        similarity = 1 - cosine(features, feature_list[index])
        similarity_percentage = similarity * 100  # Convert to percentage
        similarities.append(similarity_percentage)
    return similarities


# File upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # Recommendation
        distances, indices = recommend(features, feature_list)
        # Calculate similarity percentages
        similarities = calculate_similarity_percentage(features, feature_list, indices)

        # Show recommended images and their similarity percentages
        cols = st.columns(5)
        for col, idx, sim in zip(cols, indices[0], similarities):
            with col:
                st.image(filenames[idx])
                st.text(f"Similarity: {sim:.2f}%")
    else:
        st.header("Some error occurred in file upload")
