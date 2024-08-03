# import tensorflow as tf
# import keras
# import numpy as np
# from numpy.linalg import norm
# import os
# from tqdm import tqdm
# import pickle
#
# model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model.trainable = False
#
# model = keras.Sequential([
#     model,
#     keras.layers.GlobalMaxPooling2D()
# ])
#
# # print(model.summary())
#
#
# def extract_features(img_path, model):
#     img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
#     img_array = keras.preprocessing.image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = keras.applications.resnet.preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)
#
#     return normalized_result
#
#
# filenames = []
#
# for file in os.listdir('images'):
#     filenames.append(os.path.join('images', file))
#
# feature_list = []
#
# for file in tqdm(filenames):
#     feature_list.append(extract_features(file, model))
#
# pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
# pickle.dump(filenames, open('filenames.pkl', 'wb'))

import tensorflow as tf
import keras
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from PIL import ImageFile

# Ensure truncated images are loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = keras.Sequential([
    model,
    keras.layers.GlobalMaxPooling2D()
])


# print(model.summary())

def extract_features(img_path, model):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = keras.applications.resnet.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


filenames = []

for file in os.listdir('Myntra_Data/Images_new'):
    filenames.append(os.path.join('Myntra_Data/Images_new', file))

feature_list = []

for file in tqdm(filenames):
    try:
        features = extract_features(file, model)
        feature_list.append(features)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

pickle.dump(feature_list, open('embeddings_new.pkl', 'wb'))
pickle.dump(filenames, open('filenames_new.pkl', 'wb'))
