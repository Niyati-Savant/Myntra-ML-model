import pickle
import tensorflow
import keras
import numpy as np
from numpy.linalg import norm
import os
from sklearn.neighbors import NearestNeighbors
import cv2
import pandas as pd
pd.set_option('display.max_colwidth', None)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

feature_list = np.array(pickle.load(open('embeddings_new.pkl', 'rb')))
filenames = pickle.load(open('filenames_new.pkl', 'rb'))
csv_data = pd.read_csv('Myntra_Data/Fashion Dataset v2.csv')

model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = keras.Sequential([
    model,
    keras.layers.GlobalMaxPooling2D()
])

# print(model.summary())
img = keras.preprocessing.image.load_img('Myntra_Data/Images_new/9090955.jpg', target_size=(224, 224))
img_array = keras.preprocessing.image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = keras.applications.resnet.preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

input_img = cv2.imread('Myntra_Data/Images_new/9090955.jpg')
cv2.imshow('USER INPUT', cv2.resize(input_img, (512, 512)))
cv2.waitKey(0)

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

result_pid_img = {}
for file in indices[0]:
    filename = filenames[file]
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('OUTPUT', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)
    # Extract the base name and split to get the number part
    product_id = os.path.basename(filename).split('.')[0]
    print(f"File ID: {product_id}")
    req_row = csv_data.loc[csv_data['p_id'] == int(product_id)]
    result_img= req_row['img'].values[0]
    result_pid_img[product_id]=result_img
    print(f"Img  for {product_id}: {result_pid_img[product_id]}")


