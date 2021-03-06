import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def early_stop(array, idx, patience=1, min_delta=0.0):
    if idx <= patience:
        return False

    value = array[idx - patience]
    arr = array[idx - patience + 1:]
    check = 0
    for val in arr:
        if val - value > min_delta:
            check += 1
    if check == patience:
        return True
    else:
        return False


def transform_activation(activation):
    if activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'sigmoid':
        return tf.nn.sigmoid
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'elu':
        return tf.nn.elu
    else:
        raise Exception("Don't exists activation: " + activation)


def transform_optimizer(optimizer):
    if optimizer == 'adam':
        return tf.train.AdamOptimizer
    elif optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer
    elif optimizer == 'gd':
        return tf.train.GradientDescentOptimizer
    elif optimizer == 'momentum':
        return tf.train.MomentumOptimizer
    else:
        raise Exception("Don't exists optimizer:" + optimizer)


def transform_cell_type(cell_type):
    if cell_type == 'lstm':
        return tf.nn.rnn_cell.LSTMCell
    elif cell_type == 'gru':
        return tf.nn.rnn_cell.GRUCell
    elif cell_type == 'multirnn':
        return tf.nn.rnn_cell.MultiRNNCell
    else:
        raise Exception("Don't exists cell type:" + cell_type)


def plot_predicts(actual, predict):
    assert np.ndim(actual) == np.ndim(predict)
    ndim = np.ndim(actual)
    if ndim == 3:
        actual = actual[:, 0, -1]
        predict = predict[:, -1, -1]
    elif ndim == 2:
        actual = actual[:, 0]
        predict = predict[:, -1]

    plt.plot(actual, label='actual')
    plt.plot(predict, label='predict')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Predict')
    plt.show()


def plot_history(lines, labels):
    assert len(lines) == len(labels)
    for key, value in zip(labels, lines):
        plt.plot(value, label=key)

    plt.legend()
    plt.show()

# a = np.array([1, 2, 3, 4, 5])
# b = np.array([2, 3, 4, 5, 6])
# plot_history({'a':a, 'b':b})