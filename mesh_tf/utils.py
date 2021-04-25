import tensorflow.compat.v1 as tf
import numpy as np
import gzip
import os
import shutil
import tempfile
from six.moves import urllib


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    url = 'http://yann.lecun.com/exdb/mnist/' + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
            tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))


def dataset(images_file, labels_file):
    """Parse MNIST dataset."""

    def decode_image(img):
        # Normalize from [0, 255] to [0.0, 1.0]
        img = tf.decode_raw(img, tf.uint8)
        img = tf.cast(img, tf.float32)
        img = tf.reshape(img, [784])
        return img / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(
        images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train_data():
    """tf.data.Dataset object for MNIST training data."""
    return dataset('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')


def test_data():
    """tf.data.Dataset object for MNIST test data."""
    return dataset('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')


if __name__ == '__main__':
    # download(".", 'train-images-idx3-ubyte')
    # download(".", 't10k-images-idx3-ubyte')
    check_image_file_header('train-images-idx3-ubyte')
    check_image_file_header('t10k-images-idx3-ubyte')
    check_labels_file_header('train-labels-idx1-ubyte')
    check_labels_file_header('t10k-labels-idx1-ubyte')
