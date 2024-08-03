import tensorflow as tf
import keras
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from PIL import ImageFile
from scipy.spatial.distance import cosine

# Ensure truncated images are loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = keras.Sequential([
    model,
    keras.layers.GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = keras.applications.resnet.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Load feature vectors and filenames
feature_list = pickle.load(open('embeddings_new.pkl', 'rb'))
filenames = pickle.load(open('filenames_new.pkl', 'rb'))

def find_most_similar_image(target_image_path, feature_list, filenames, model):
    target_features = extract_features(target_image_path, model)
    similarities = []

    for features in feature_list:
        similarity = 1 - cosine(target_features, features)
        similarities.append(similarity)

    most_similar_index = np.argmax(similarities)
    most_similar_image = filenames[most_similar_index]
    most_similar_score = similarities[most_similar_index] * 100  # Convert to percentage

    return most_similar_image, most_similar_score

# Example usage
target_image_path = 'Myntra_Data/Images_new/test.jpg'
most_similar_image, most_similar_score = find_most_similar_image(target_image_path, feature_list, filenames, model)

print(f"Most similar image: {most_similar_image}")
print(f"Similarity percentage: {most_similar_score:.2f}%")
