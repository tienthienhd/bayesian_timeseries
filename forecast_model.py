import time

import tensorflow as tf
import numpy as np
import json
from ed_model import EDModel


class AnnModel(object):
    def __init__(self, model_dir, model_ed_dir):
        tf.reset_default_graph()

        self.model_dir = model_dir
        self.model = None
        self.ed_model = EDModel(model_ed_dir)
        self.ed_model.restore()
        self.sliding = self.ed_model.sliding_encoder
        self.n_dim = self.ed_model.layer_sizes_ed[-1]

        self.sess = tf.Session()

    def _parse_params(self, params):
        params['n_dim'] = self.n_dim
        self.params = params
        self.layer_sizes_ann = params['layer_sizes_ann']
        self.dropout = params['dropout']
        self.patience = params['patience']

        ac = params['activation']
        if ac == 'tanh':
            self.activation = tf.nn.tanh
        elif ac == 'sigmoid':
            self.activation = tf.nn.sigmoid
        elif ac == 'relu':
            self.activation = tf.nn.relu
        else:
            raise Exception("Don't have activation:" + ac)

        optimizer = params['optimizer']
        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer
        else:
            raise Exception("Don't have optimizer:" + optimizer)

        self.learning_rate = params['learning_rate']

    def build_model(self, params):
        self._parse_params(params)

        # Define input
        self.x = tf.placeholder(tf.float32, (None, self.sliding, self.n_dim), 'x')
        self.y = tf.placeholder(tf.float32, (None, 1), 'y')

        # Define architecture
        with tf.variable_scope('architecture'):
            net = self.x
            for i, units in enumerate(self.layer_sizes_ann):
                net = tf.layers.dense(net, units=units, activation=self.activation, name='layer_'+str(i))
                net = tf.layers.dropout(net, rate=self.dropout, name='dropout_l'+str(i))
            net = tf.layers.dense(net, units=1, activation=self.activation) # combine data dimension
            net = tf.reshape(net, (-1, self.sliding))
            output = tf.layers.dense(net, units=1) # predict next value
            self.preds = tf.identity(output, 'predict')


        # Define loss
        with tf.variable_scope('loss_and_metrics'):
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y, output)))
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.y, output)))
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.y, output))))

        # Define train operation
        with tf.variable_scope('optimizer'):
            self.train_op = self.optimizer(self.learning_rate).minimize(self.loss)

        tf.add_to_collection('params', self.x)
        tf.add_to_collection('params', self.y)
        tf.add_to_collection('params', self.preds)
        tf.add_to_collection('params', self.loss)
        tf.add_to_collection('params', self.mae)
        tf.add_to_collection('params', self.rmse)
        tf.add_to_collection('params', self.train_op)

        # Initial variables
        self.sess.run(tf.global_variables_initializer())

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.model_dir + '/ann_model')

        with open(self.model_dir + "/hyper_params.json", 'w') as f:
            json.dump(self.params, f)

    def restore(self):
        saver = tf.train.import_meta_graph(self.model_dir + '/ann_model.meta')
        saver.restore(self.sess, self.model_dir+'/ann_model')

        params = tf.get_collection('params')
        self.x = params[0]
        self.y = params[1]
        self.preds = params[2]
        self.loss = params[3]
        self.mae = params[4]
        self.rmse = params[5]
        self.train_op = params[-1]

        self.params = json.load(open(self.model_dir + '/hyper_params.json', 'r'))
        self.layer_sizes_ann = self.params['layer_sizes_ann']
        self.activation = self.params['activation']
        self.optimizer = self.params['optimizer']
        self.learning_rate = self.params['learning_rate']
        self.dropout = self.params['dropout']

    def _compute_features(self, x):
        features = self.ed_model.get_features(x)
        # features = np.mean(features, axis=2)
        # print('features shape:', features.shape)
        return features

    def train(self, x, y, validation_split=0.2, batch_size=32, epochs=1, verbose=1):
        x = self._compute_features(x)
        n_train = int(len(y) * (1 - validation_split))
        x_train = x[:n_train]
        y_train = y[:n_train]

        x_val = x[n_train:]
        y_val = y[n_train:]

        n_batches = int(n_train / batch_size)
        if n_train % batch_size != 0:
            n_batches += 1

        history = {
            'loss': [],
            'mae': [],
            'rmse': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': []
        }

        for e in range(epochs):
            start_epoch_time = time.time()
            loss = 0.0
            mae = 0.0
            rmse = 0.0
            for b in range(n_batches):
                xb = x_train[b * batch_size: (b + 1) * batch_size]
                yb = y_train[b * batch_size: (b + 1) * batch_size]

                l, m, r, _ = self.sess.run([self.loss, self.mae, self.rmse, self.train_op], feed_dict={
                    self.x: xb,
                    self.y: yb
                })
                loss += l
                mae += m
                rmse += np.square(r)  # mean square error
            loss /= n_batches
            mae /= n_batches
            rmse /= n_batches  # mean square error
            rmse = np.sqrt(rmse)
            history['loss'].append(loss)
            history['mae'].append(mae)
            history['rmse'].append(rmse)

            val_loss, val_mae, val_rmse = self.sess.run([self.loss, self.mae, self.rmse], feed_dict={
                self.x: x_val,
                self.y: y_val
            })
            val_mae = val_mae
            val_rmse = val_rmse
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)

            epoch_time = time.time() - start_epoch_time
            if verbose > 0:
                print(
                    "Epoch {}/{}: time={:.2f}s, loss={:.5f}, mae={:.5f}, rmse={:.5f}, val_loss={:.5f}, val_mae={:.5f}, val_rmse={:.5f}".format(
                        e + 1, epochs, epoch_time,
                        loss, mae, rmse, val_loss, val_mae, val_rmse))
            if self._early_stop(history['val_loss'], e, patience=self.patience):
                print('Early stop at epoch', (e + 1))
                break
            if np.isnan(loss):
                break
        return history

    @staticmethod
    def _early_stop(array, idx, patience=1, min_delta=0.0):
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

    def eval(self, x, y):
        features = self._compute_features(x)
        return self.sess.run([self.loss, self.mae, self.rmse], {self.x: features, self.y: y})

    def predict(self, x):
        features = self._compute_features(x)
        return self.sess.run(self.preds, {self.x: features})




from dataset import GgTraceDataSet2, split_data


params = {
    'layer_sizes_ann': [64],
    'activation': 'sigmoid',
    'optimizer': 'rmsprop',
    'dropout': 0.05,
    'batch_size': 8,
    'learning_rate': 0.001,
    'epochs': 200,
    'patience': 5,
}

dataset = GgTraceDataSet2('datasets/5.csv', 8, 1)
params['n_dim'] = dataset.n_dim
data = dataset.get_data_forecast()
train, test = split_data(data, test_size=0.2)
x_train = train[0]
y_train = train[1].reshape((-1, 1))
x_test = test[0]
y_test = test[1].reshape((-1, 1))

print(x_train.shape, y_train.shape)

model = AnnModel('logs/test_ann', 'logs/test')
model.build_model(params)
model.train(x_train, y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            verbose=1)

model.save()
# model.restore()
print(model.params)
print(model.eval(x_test, y_test))
print(model.eval(x_test, y_test))
print(model.eval(x_test, y_test))
print(model.eval(x_test, y_test))
print(model.eval(x_test, y_test))
print(model.eval(x_test, y_test))
preds = model.predict(x_test)


import matplotlib.pyplot as plt

plt.plot(y_test)
plt.plot(preds)
plt.legend(['actual', 'predict'])

plt.show()