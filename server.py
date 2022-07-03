# Load image uploaded by user
# Preprocessing
# Predict
# Return result
from fastapi import FastAPI, File, UploadFile
#from prediction import predict, read_image
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
import numpy as np
from io import BytesIO
from PIL import Image
from retinaface import RetinaFace
import cv2
import joblib

app = FastAPI()

global pca, model
classes = ['Thu', 'Chau', 'LeVan', 'VAnh', 'Linh', 'Thang',
           'Kien', 'Van', 'Tan', 'Quan', 'Tuan', 'Truong',
           'XAnh', 'Hieu', 'Hung', 'HDuc', 'Duc', 'VDuc', 'Unknown']

with open("pca.pkl", 'rb') as f1:
    pca = joblib.load(f1)


with open("model4.pkl", 'rb') as f2:
    model = joblib.load(f2)

print(model)


@app.post("/predict-tf/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    res = RetinaFace.detect_faces(file.filename)
    axis = res['face_1']['facial_area']

    image = Image.open(BytesIO(await file.read()))
    image = image[axis[1]:axis[3], axis[0]:axis[2]]

    model = tf.keras.models.load_model("model4.h5")

    image = img_to_array(image.resize((299, 299)))
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    res = model.predict(image)

    response = {"class": res.argmax()}

    return response


@app.post("/predict-svm/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    res = RetinaFace.detect_faces(file.filename)
    axis = res['face_1']['facial_area']

    image = Image.open(BytesIO(await file.read()))
    image = image[axis[1]:axis[3], axis[0]:axis[2]]
    image = Image.open(BytesIO(await file.read()))

    image = img_to_array(image.resize((299, 299)))
    image = np.expand_dims(image, 0)
    image = np.mean(image, axis=3)
    x = image.flatten()
    x = x.reshape(1, -1)

    x = pca.transform(x)

    res = model.predict(x)

    return classes[res[0]]
