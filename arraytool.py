import zmq
import numpy as np


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
