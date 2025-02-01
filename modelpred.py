import cv2
import tensorflow as tf
from tensorflow.python.keras.models import load_model

CATEGORIES = ["normal", "tumor"]


def prepare(filepath):
    IMG_SIZE = 256  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = load_model("6CNN.model")

prediction = model.predict([prepare('test2.jpg')])

print(prediction)
print(CATEGORIES[int(prediction[0][0])])


