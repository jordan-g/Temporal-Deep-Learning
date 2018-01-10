import numpy as np
import torch

def load_mnist_data(n_examples, n_test_examples, validation=True):
    x_set      = np.load("mnist/x_train.npy").astype(np.float32)
    t_set      = np.load("mnist/t_train.npy").astype(np.float32)
    if validation:
        x_test_set = x_set[:, n_examples:n_examples+n_test_examples]
        t_test_set = t_set[:, n_examples:n_examples+n_test_examples]
    else:
        x_test_set = np.load("mnist/x_test.npy").astype(np.float32)[:, :n_test_examples]
        t_test_set = np.load("mnist/t_test.npy").astype(np.float32)[:, :n_test_examples]

    x_set = x_set[:, :n_examples]
    t_set = t_set[:, :n_examples]

    return x_set, t_set, x_test_set, t_test_set

def load_cifar10_data(n_examples, n_test_examples, validation=True):
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    data_batches = []
    labels = []

    for i in range(5):
        dictionary = unpickle("cifar-10/data_batch_{}".format(i+1))
        data_batches.append(dictionary[b'data'])
        labels += dictionary[b'labels']

    x_set = np.concatenate(data_batches, axis=0).T.astype(np.float32)/255.0
    t_set = np.zeros((10, len(labels))).astype(np.float32)
    for i in range(len(labels)):
        t_set[labels[i], i] = 1

    if validation:
        x_test_set = x_set[:, n_examples:n_examples+n_test_examples]
        t_test_set = t_set[:, n_examples:n_examples+n_test_examples]
    else:
        dictionary = unpickle("cifar-10/test_batch")
        labels = dictionary[b'labels']
        x_test_set = dictionary[b'data'].T.astype(np.float32)[:, :n_test_examples]/255.0
        t_test_set = np.zeros((10, n_test_examples)).astype(np.float32)
        for i in range(n_test_examples):
            t_test_set[labels[i], i] = 1

    x_set = x_set[:, :n_examples]
    t_set = t_set[:, :n_examples]

    return x_set, t_set, x_test_set, t_test_set