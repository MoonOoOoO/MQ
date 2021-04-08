#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import zmq
import time
import numpy as np

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


while True:
    #  Wait for next request from client
    message = recv_array(socket)
    print("Received request: %s" % str(message))

    #  Send reply back to client
    socket.send(b"World")
