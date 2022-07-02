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
import pickle
import joblib

app = FastAPI()

global pca, model

with open("pca.pkl", 'rb') as f1:
    pca = joblib.load(f1)


with open("model4.pkl", 'rb') as f2:
    model = joblib.load(f2)

print (model)

@app.post("/predict-tf/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    image = Image.open(BytesIO(await file.read()))

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

    image = Image.open(BytesIO(await file.read()))

    image = img_to_array(image.resize((299, 299)))
    image = np.expand_dims(image, 0)
#     image = preprocess_input(image)
    image = np.mean(image, axis=3)
    x = image.flatten()
    x = x.reshape(1, -1)


    x = pca.transform(x)

#     x = x.reshape(-1, 1)

    res = model.predict(x)

    

    return res[0]
