#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from concurrent.futures import ThreadPoolExecutor

context = zmq.Context()

#  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

IMAGE_PATH = "elephant.jpg"


def preprocessing(img_path):
    x = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)
    return x


def send_array(s, arr, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(arr.dtype),
        shape=arr.shape
    )
    s.send_json(md, flags | zmq.SNDMORE)
    return s.send(arr, flags, copy=copy, track=track)


#  Do 10 requests, waiting each time for a response
for request in range(10):
    tic = time.time()
    img = preprocessing(IMAGE_PATH)

    img = np.array(img)
    # socket.send(img)
    send_array(socket, img)

    #  Get the reply.
    message = socket.recv()
    # print(message.decode('utf-8'));
    print((time.time() - tic) * 1000)
