import zmq
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


def send_array(s, arr, flags=0, copy=True, track=False):
    arr = np.array(arr)
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(arr.dtype),
        shape=arr.shape
    )
    s.send_json(md, flags | zmq.SNDMORE)
    return s.send(arr, flags, copy=copy, track=track)


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
    #  Send reply back to client
    # socket.send(bytes(' '.join(map(str, result)), 'utf-8'))
    send_array(socket, result)
