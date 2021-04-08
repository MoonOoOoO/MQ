#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import array
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.saved_model import tag_constants
from concurrent.futures import ThreadPoolExecutor

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

IMAGE_PATH = "elephant.jpg"


def preprocessing(img_path):
    x = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def send_array(s, arr, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(arr.dtype),
        shape=arr.shape,
    )
    s.send_json(md, flags | zmq.SNDMORE)
    return s.send(arr, flags, copy=copy, track=track)


a = np.array([0, 1, 2, 3, 4, 5])

#  Do 10 requests, waiting each time for a response
for request in range(10):
    img = preprocessing(IMAGE_PATH)
    print("Sending request %s …" % request)
    send_array(socket, img)

    #  Get the reply.
    message = socket.recv()
    print("Received reply %s [ %s ]" % (request, message))
