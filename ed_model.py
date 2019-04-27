import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json


class EDModel(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.params = None
        self.sliding_encoder = 0
        self.sliding_decoder = 0
        self.layer_sizes_ed = []
        self.n_dim = 0
        self.activation = tf.nn.relu
        self.keep_probs = 1
        # self.variational_recurrent = False
        self.cell_type = tf.nn.rnn_cell.LSTMCell
        self.optimizer = tf.train.AdamOptimizer
        self.learning_rate = 0.001
        self.patience = 2

        self.x_e = None
        self.x_d = None
        self.y_d = None
        self.feature = None
        self.preds = None
        self.loss = 0
        self.mae = 0
        self.rmse = 0
        self.train_op = None

        tf.reset_default_graph()
        self.sess = tf.Session()

    def _parse_params(self, params):
        self.sliding_encoder = int(params['sliding_encoder'])
        self.sliding_decoder = int(params['sliding_decoder'])
        self.layer_sizes_ed = params['layer_sizes_ed']
        self.n_dim = int(params['n_dim'])
        self.keep_probs = params['keep_probs']
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

        cell_type = params['cell_type']
        if cell_type == 'lstm':
            self.cell_type = tf.nn.rnn_cell.LSTMCell
        elif cell_type == 'gru':
            self.cell_type = tf.nn.rnn_cell.GRUCell
        else:
            raise Exception("Don't hava cell type:" + cell_type)

        optimizer = params['optimizer']
        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer
        else:
            raise Exception("Don't have optimizer:" + optimizer)

        self.learning_rate = params['learning_rate']

    def _create_block_rnn(self, inputs, state=None):
        cells = []
        for i, units in enumerate(self.layer_sizes_ed):
            # create cell
            cell = self.cell_type(num_units=units, activation=self.activation, name="layer_" + str(i))

            # Wrap cell with dropout
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_probs,
                                                 output_keep_prob=self.keep_probs,
                                                 state_keep_prob=self.keep_probs,
                                                 variational_recurrent=True,
                                                 input_size=self.n_dim if i == 0 else self.layer_sizes_ed[i - 1],
                                                 dtype=tf.float32)
            cells.append(cell)
        # Multi cell layer
        cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        if state is None:
            output, state = tf.nn.dynamic_rnn(cells, inputs=inputs, dtype=tf.float32)
        else:
            output, state = tf.nn.dynamic_rnn(cells, inputs=inputs, initial_state=state, dtype=tf.float32)

        return output, state

    def build_model(self, params):
        self._parse_params(params)

        # Placeholder input
        self.x_e = tf.placeholder(tf.float32, (None, self.sliding_encoder, self.n_dim), 'x_e')
        self.x_d = tf.placeholder(tf.float32, (None, self.sliding_decoder, self.n_dim), 'x_d')
        self.y_d = tf.placeholder(tf.float32, (None, self.sliding_decoder, self.n_dim), 'y_d')

        # add to collection
        tf.add_to_collection('params', self.x_e)
        tf.add_to_collection('params', self.x_d)
        tf.add_to_collection('params', self.y_d)

        # Encoder
        with tf.variable_scope('encoder'):
            out_e, state_e = self._create_block_rnn(self.x_e, state=None)

            self.feature = tf.identity(out_e, 'feature')
            tf.add_to_collection('params', self.feature)

        with tf.variable_scope('decoder'):
            out_d, state_d = self._create_block_rnn(self.x_d, state_e)

            preds = tf.layers.dense(out_d, units=1)
            self.preds = tf.identity(preds, 'predict_decoder')
            tf.add_to_collection('params', self.preds)

        with tf.variable_scope('loss_and_metrics'):
            self.loss = tf.losses.mean_squared_error(self.y_d, preds)
            self.mae = tf.reduce_mean(tf.abs(self.y_d - preds))#tf.metrics.mean_absolute_error(self.y_d, preds)
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.y_d - preds))) #tf.metrics.root_mean_squared_error(self.y_d, preds)

            tf.add_to_collection('params', self.loss)
            tf.add_to_collection('params', self.mae)
            tf.add_to_collection('params', self.rmse)

        with tf.variable_scope('optimizer'):
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 2000, 0.96, staircase=True)

            self.train_op = self.optimizer(learning_rate).minimize(self.loss, global_step=global_step)
            tf.add_to_collection('params', self.train_op)

        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.model_dir+'/ed_model')

        with open(self.model_dir + "/hyper_params.json", 'w') as f:
            json.dump(self.params, f)

    def restore(self):
        saver = tf.train.import_meta_graph(self.model_dir + '.meta')
        saver.restore(self.sess, self.model_dir)

        params = tf.get_collection('params')
        self.x_e = params[0]
        self.x_d = params[1]
        self.y_d = params[2]
        self.feature = params[3]
        self.preds = params[4]
        self.loss = params[5]
        self.mae = params[6]
        self.rmse = params[7]
        self.train_op = params[-1]

        self.params = json.load(open(self.model_dir + '/hyper_params.json', 'r'))
        self.params = self.params
        self.sliding_encoder = self.params['sliding_encoder']
        self.sliding_decoder = self.params['sliding_decoder']
        self.layer_sizes_ed = self.params['layer_sizes_ed']
        # self.layer_sizes_ann = self.params['layer_sizes_ann']
        self.n_dim = self.params['n_dim']
        self.activation = self.params['activation']
        self.optimizer = self.params['optimizer']
        self.learning_rate = self.params['learning_rate']
        self.keep_probs = self.params['keep_probs']
        cell_type = self.params['cell_type']
        if cell_type == 'lstm':
            self.cell_type = tf.nn.rnn_cell.LSTMCell
        elif cell_type == 'gru':
            self.cell_type = tf.nn.rnn_cell.GRUCell

    def train(self, x, y, validation_split=0.2, batch_size=32, epochs=1, verbose=1):
        n_train = int(len(y) * (1 - validation_split))
        xe_train = x[0][:n_train]
        xd_train = x[1][:n_train]
        yd_train = y[:n_train]

        xe_val = x[0][n_train:]
        xd_val = x[1][n_train:]
        yd_val = y[n_train:]

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
                xe = xe_train[b * batch_size: (b + 1) * batch_size]
                xd = xd_train[b * batch_size: (b + 1) * batch_size]
                yd = yd_train[b * batch_size: (b + 1) * batch_size]

                try:
                    l, m, r, _ = self.sess.run([self.loss, self.mae, self.rmse, self.train_op], feed_dict={
                        self.x_e: xe,
                        self.x_d: xd,
                        self.y_d: yd
                    })
                except ValueError:
                    print("============>Exception: " + xd_train.shape)
                loss += l
                mae += m
                rmse += np.square(r) # mean square error
            loss /= n_batches
            mae /= n_batches
            rmse /= n_batches # mean square error
            rmse = np.sqrt(rmse)
            history['loss'].append(loss)
            history['mae'].append(mae)
            history['rmse'].append(rmse)

            val_loss, val_mae, val_rmse = self.sess.run([self.loss, self.mae, self.rmse], feed_dict={
                self.x_e: xe_val,
                self.x_d: xd_val,
                self.y_d: yd_val
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

    def get_features(self, xe):
        return self.sess.run(self.feature, feed_dict={
            self.x_e: xe,
        })

    def eval(self, x, y):
        xe = x[0]
        xd = x[1]
        yd = y
        return self.sess.run([self.loss, self.mae, self.rmse], feed_dict={
            self.x_e: xe,
            self.x_d: xd,
            self.y_d: yd
        })

    def predict(self, x):
        xe = x[0]
        xd = x[1]
        return self.sess.run(self.preds, feed_dict={
            self.x_e: xe,
            self.x_d: xd,
        })


# from dataset import GgTraceDataSet, split_data
#
# params_example = {
#     'sliding_encoder': 4,
#     'sliding_decoder': 2,
#     'layer_sizes_ed': [32],
#     # 'layer_sizes_ann': [32, 4],
#     'activation': 'tanh',
#     'optimizer': 'adam',
#     # 'n_dim': 1,
#     'input_keep_prob': 0.5,
#     'output_keep_prob': 0.5,
#     'state_keep_prob': 0.5,
#     'batch_size': 1,
#     'learning_rate': 0.001,
#     'epochs': 100,
#     'cell_type': 'lstm',
# }
#
# dataset = GgTraceDataSet('datasets/5.csv',
#                          params_example['sliding_encoder'],
#                          params_example['sliding_decoder'])
# params_example['n_dim'] = dataset.n_dim
# data = dataset.get_data()
# train, test = split_data(data)
#
# model = EDModel('logs/test')
# model.build_model(params_example)
# history = model.train([train[0], train[1]], train[2],
#             batch_size=params_example['batch_size'],
#             epochs=params_example['epochs'])
#
# from utils import plot_history, plot_predicts
#
# plot_history((history['loss'], history['val_loss']), ('loss', 'val_loss'))
#
# print(model.eval([test[0], test[1]], test[2]))
#
# print(model.get_features(test[0])[0, :2, :2])
# print(model.get_features(test[0])[0, :2, :2])
#
# preds = model.predict(x=[test[0], test[1]])
# plot_predicts(actual=test[2], predict=preds)