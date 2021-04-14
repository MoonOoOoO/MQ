import cv2
import zmq
import time
import threading
import numpy as np
import tensorflow as tf
from queue import Empty, Queue
from flask import Flask, request as flask_request
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from arraytool import send_array, recv_array

app = Flask(__name__)
context = zmq.Context()

#  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

IMAGE_PATH = "elephant.jpg"

BATCH_SIZE = 64
BATCH_TIMEOUT = 0.6
CHECK_INTERVAL = 0.01

requests_queue = Queue()


def preprocessing(img_path):
    x = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)
    return x


def prepare_image(raw_image):
    x = cv2.resize(raw_image, dsize=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def form_batch(requests_batch):
    batched_input = np.zeros((len(requests_batch), 224, 224, 3), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=BATCH_SIZE * 2) as executor:
        results = [executor.submit(prepare_image, img['input']) for img in requests_batch]
    for i, x in zip(range(BATCH_SIZE), results):
        batched_input[i, :] = x.result()
    return np.array(tf.constant(batched_input))


def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (
                len(requests_batch) > BATCH_SIZE or
                (len(requests_batch) > 0 and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue
        batched_input = form_batch(requests_batch)
        send_array(socket, batched_input)
        # batch_outputs = model.predict(batched_input)
        # message = socket.recv()
        message = recv_array(socket)

        for request, output in zip(requests_batch, message):
            request['output'] = str(output)


# #  Do 10 requests, waiting each time for a response
# for request in range(10):
#     tic = time.time()
#     img = preprocessing(IMAGE_PATH)
#
#     img = np.array(img)
#     # socket.send(img)
#     send_array(socket, img)
#
#     #  Get the reply.
#     message = socket.recv()
#     # print(message.decode('utf-8'));
#     print((time.time() - tic) * 1000)

threading.Thread(target=handle_requests_by_batch).start()


@app.route('/predict', methods=['POST'])
def predict():
    if flask_request.method == 'POST':
        f = flask_request.files['file']
        if f:
            img_from_url = np.fromstring(f.read(), np.uint8)
            img_from_url = cv2.imdecode(img_from_url, cv2.IMREAD_COLOR)
            req = {'input': img_from_url, 'time': time.time()}
            requests_queue.put(req)
            while 'output' not in req:
                time.sleep(CHECK_INTERVAL)
            return {'predictions': req['output']}


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)
