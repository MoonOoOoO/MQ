import zmq
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions

from arraytool import send_array, recv_array

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

model = ResNet50(weights="imagenet")

while True:
    #  Wait for next request from client
    message = recv_array(socket)
    # model prediction the received numpy array
    pred = model.predict(message)
    result = decode_predictions(pred)
    #  Send the result back to client
    send_array(socket, result)
