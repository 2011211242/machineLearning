""""Functions for reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import numpy as np

def _read32(bytestream):
    '''Read a int32 from bytestream'''
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(work_directory, filename):
    """Extract the images into a 4D uint8 np array [index, y, x, depth]."""
    print("Extracting", filename)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        raise IOError(
                "file %s doesn't exist" 
                %(filepath))

    with gzip.open(filepath) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes, 1))
    labels_one_hot.flat[index_offset + labels_dense] = 1
    return labels_one_hot

def extract_labels(work_directory, filename, one_hot=False):
    """Extract the labels into a 1D uint8 np array [index]."""
    print("Extracting", filename)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        raise IOError(
                "file %s doesn't exist" 
                %(filepath))

    with gzip.open(filepath) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))

        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels

def wrap_data(images, labels):
    """Wrap the data in zip"""
    images = images.reshape(images.shape[0], 
            images.shape[1] * images.shape[2], 1)
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0/255.0)
    return zip(images, labels)

def read_data_sets(train_dir, one_hot=False):
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    train_images = extract_images(train_dir, TRAIN_IMAGES)
    train_labels = extract_labels(train_dir, TRAIN_LABELS, one_hot=True)
    test_images = extract_images(train_dir, TEST_IMAGES)
    test_labels = extract_labels(train_dir, TEST_LABELS, one_hot=True)

    VALIDATION_SIZE = 10000
    data = wrap_data(train_images, train_labels)
    validation_data = data[:VALIDATION_SIZE]
    training_data = data[VALIDATION_SIZE:]
    test_data = wrap_data(test_images, test_labels)
    return training_data, validation_data, test_data

if __name__ == "__main__":
    training_images, validation_data, test_data = \
            read_data_sets("MNIST_data", one_hot = True)
    (x, y) = training_images[0]
    print(x.shape)
    print(y.shape)
