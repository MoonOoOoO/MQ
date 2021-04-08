#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import zmq
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


def recv_array(s, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = s.recv_json(flags=flags)
    msg = s.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    arr = np.frombuffer(buf, dtype=md['dtype'])
    return arr.reshape(md['shape'])


model = ResNet50(weights="imagenet")

while True:
    #  Wait for next request from client
    message = recv_array(socket)
    # message = socket.recv()

    pred = model.predict(message)
    result = decode_predictions(pred)
    print("Prediction result: %s" % str(result))

    #  Send reply back to client
    socket.send(result)
