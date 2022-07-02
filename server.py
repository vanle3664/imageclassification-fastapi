# Load image uploaded by user
# Preprocessing
# Predict
# Return result
from fastapi import FastAPI, File, UploadFile
#from prediction import predict, read_image
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
from io import BytesIO
from PIL import Image


app = FastAPI()


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    image = Image.open(BytesIO(await file.read()))

    model = tf.keras.models.load_model("../CV/model4.h5")

    image = img_to_array(image.resize((299, 299)))
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    res = model.predict(image)

    response = {"class": res.argmax()}

    return response
