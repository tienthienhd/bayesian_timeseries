import time

import tensorflow as tf
import numpy as np
import json
import utils

tf.logging.set_verbosity(tf.logging.ERROR)


class Model(object):
    def __init__(self, model_dir, params=None):
        self.model_dir = model_dir
        tf.reset_default_graph()
        self._sess = tf.Session()

        if params is not None:
            self._params = params
            self._sle = int(params['sliding_encoder'])
            self._sld = int(params['sliding_decoder'])
            self._layer_sizes_ed = params['layer_sizes_ed']
            self._layer_sizes_f = params['layer_sizes_f']
            self._n_dim = int(params['n_dim'])
            self._keep_probs = params['keep_probs']
            self._dropout = params['dropout']
            self._patience = params['patience']
            self._learning_rate = params['learning_rate']

            self._activation = utils.transform_activation(params['activation'])
            self._optimizer = utils.transform_optimizer(params['optimizer'])
            self._cell_type = utils.transform_cell_type(params['cell_type'])

            self.build_model()
        else:
            self.restore()

    def build_model(self):

        # input
        self._xe = tf.placeholder(tf.float32, (None, self._sle, self._n_dim), 'ex')
        self._xd = tf.placeholder(tf.float32, (None, self._sld, self._n_dim), 'dx')
        self._yd = tf.placeholder(tf.float32, (None, self._sld, 1), 'dy')
        self._yf = tf.placeholder(tf.float32, (None, 1), 'fy')

        # encoder
        with tf.variable_scope('encoder'):
            out_e, state_e = self._create_block_rnn(self._xe, state=None)

        # decoder
        with tf.variable_scope('decoder'):
            out_d, state_d = self._create_block_rnn(self._xd, state=state_e)
            self._pred_d = tf.layers.dense(out_d, units=1)

        # forecaster
        with tf.variable_scope('forecaster'):
            out_e = tf.stop_gradient(out_e, 'features')
            out_f = self._create_block_dense(out_e)
            self._pred_f = tf.layers.dense(out_f, units=1)

        # loss
        with tf.variable_scope('loss'):
            self._loss_ed = tf.losses.mean_squared_error(self._yd, self._pred_d)
            self._loss_f = tf.losses.mean_squared_error(self._yf, self._pred_f)

        # metrics
        with tf.variable_scope('metrics'):
            self._mae_ed = tf.reduce_mean(tf.abs(tf.subtract(self._yd, self._pred_d)))
            self._rmse_ed = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self._yd, self._pred_d))))

            self._mae_f = tf.reduce_mean(tf.abs(tf.subtract(self._yf, self._pred_f)))
            self._rmse_f = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self._yf, self._pred_f))))

        # optimizer
        with tf.variable_scope('optimizer'):
            self._train_op_ed = self._optimizer(self._learning_rate).minimize(self._loss_ed)
            self._train_op_f = self._optimizer(self._learning_rate).minimize(self._loss_f)

        # initial variable
        self._sess.run(tf.global_variables_initializer())

    def save(self):
        # add to collection
        tf.add_to_collection('params', self._xe)
        tf.add_to_collection('params', self._xd)
        tf.add_to_collection('params', self._yd)
        tf.add_to_collection('params', self._yf)
        tf.add_to_collection('params', self._pred_d)
        tf.add_to_collection('params', self._pred_f)
        tf.add_to_collection('params', self._loss_ed)
        tf.add_to_collection('params', self._loss_f)
        tf.add_to_collection('params', self._mae_ed)
        tf.add_to_collection('params', self._mae_f)
        tf.add_to_collection('params', self._rmse_ed)
        tf.add_to_collection('params', self._rmse_f)
        tf.add_to_collection('params', self._train_op_ed)
        tf.add_to_collection('params', self._train_op_f)

        saver = tf.train.Saver()
        saver.save(self._sess, self.model_dir + '/model')

        if hasattr(self, 'params'):
            with open(self.model_dir + "/hyper_params.json", 'w') as f:
                json.dump(self._params, f)

    def restore(self):
        saver = tf.train.import_meta_graph(self.model_dir + '/model.meta')
        saver.restore(self._sess, self.model_dir + '/model')
        params = tf.get_collection('params')
        self._xe = params[0]
        self._xd = params[1]
        self._yd = params[2]
        self._yf = params[3]
        self._pred_d = params[4]
        self._pred_f = params[5]
        self._loss_ed = params[6]
        self._loss_f = params[7]
        self._mae_ed = params[8]
        self._mae_f = params[9]
        self._rmse_ed = params[10]
        self._rmse_f = params[11]
        self._train_op_ed = params[12]
        self._train_op_f = params[13]

        params = json.load(open(self.model_dir + '/hyper_params.json', 'r'))
        self._params = params
        self._sle = int(params['sliding_encoder'])
        self._sld = int(params['sliding_decoder'])
        self._layer_sizes_ed = params['layer_sizes_ed']
        self._layer_sizes_f = params['layer_sizes_f']
        self._n_dim = int(params['n_dim'])
        self._keep_probs = params['keep_probs']
        self._dropout = params['dropout']
        self._patience = params['patience']
        self._learning_rate = params['learning_rate']

        self._activation = utils.transform_activation(params['activation'])
        self._optimizer = utils.transform_optimizer(params['optimizer'])
        self._cell_type = utils.transform_cell_type(params['cell_type'])

    def train_ed(self, x, y, validation_split=0.2, batch_size=32, epochs=1, verbose=1):
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
            for b in range(n_batches):
                xe = xe_train[b * batch_size: (b + 1) * batch_size]
                xd = xd_train[b * batch_size: (b + 1) * batch_size]
                yd = yd_train[b * batch_size: (b + 1) * batch_size]

                input_feed = {
                    self._xe: xe,
                    self._xd: xd,
                    self._yd: yd
                }
                output_feed = [self._loss_ed, self._mae_ed, self._train_op_ed]

                l, m, _ = self._sess.run(output_feed, input_feed)

                loss += l
                mae += m
            loss /= n_batches
            mae /= n_batches
            history['loss'].append(loss)
            history['mae'].append(mae)

            val_loss, val_mae = self._sess.run([self._loss_ed, self._mae_ed],
                                              {
                                                  self._xe: xe_val,
                                                  self._xd: xd_val,
                                                  self._yd: yd_val
                                              })
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)

            epoch_time = time.time() - start_epoch_time
            if verbose > 0:
                print(
                    "Epoch {}/{}: time={:.2f}s, loss={:.5f}, mae={:.5f}, val_loss={:.5f}, val_mae={:.5f}".format(
                        e + 1, epochs, epoch_time,
                        loss, mae, val_loss, val_mae))
            if utils.early_stop(history['val_loss'], e, patience=self._patience):
                print('Early stop at epoch', (e + 1))
                break
            if np.isnan(loss):
                break
        return history

    def train_f(self, x, y, validation_split=0.2, batch_size=0.2, epochs=1, verbose=1):
        n_train = int(len(y) * (1 - validation_split))
        xe_train = x[:n_train]
        yf_train = y[:n_train]

        xe_val = x[n_train:]
        yf_val = y[n_train:]

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
            for b in range(n_batches):
                xe = xe_train[b * batch_size: (b + 1) * batch_size]
                yf = yf_train[b * batch_size: (b + 1) * batch_size]

                input_feed = {
                    self._xe: xe,
                    self._yf: yf
                }
                output_feed = [self._loss_f, self._mae_f, self._train_op_f]
                l, m, _ = self._sess.run(output_feed, input_feed)

                loss += l
                mae += m
            loss /= n_batches
            mae /= n_batches
            history['loss'].append(loss)
            history['mae'].append(mae)

            val_loss, val_mae = self._sess.run([self._loss_f, self._mae_f],
                                              {
                                                  self._xe: xe_val,
                                                  self._yf: yf_val
                                              })
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            epoch_time = time.time() - start_epoch_time
            if verbose > 0:
                print(
                    "Epoch {}/{}: time={:.2f}s, loss={:.5f}, mae={:.5f}, val_loss={:.5f}, val_mae={:.5f}".format(
                        e + 1, epochs, epoch_time,
                        loss, mae, val_loss, val_mae))
            if utils.early_stop(history['val_loss'], e, patience=self._patience):
                print('Early stop at epoch', (e + 1))
                break
            if np.isnan(loss):
                break
        return history

    def eval_ed(self, x, y):
        return self._sess.run([self._loss_ed, self._mae_ed], {
            self._xe: x[0],
            self._xd: x[1],
            self._yd: y
        })

    def eval_f(self, x, y):
        return self._sess.run([self._loss_f, self._mae_f], {
            self._xe: x,
            self._yf: y
        })

    def predict_ed(self, x):
        return self._sess.run(self._pred_d, {self._xe: x[0], self._xd: x[1]})

    def predict_f(self, x):
        return self._sess.run(self._pred_f, {self._xe: x})

    def _create_block_rnn(self, inputs, state=None):
        cells = []
        for i, units in enumerate(self._layer_sizes_ed):
            # create cell
            cell = self._cell_type(num_units=units, activation=self._activation, name="layer_" + str(i))

            # Wrap cell with dropout
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self._keep_probs,
                                                 output_keep_prob=self._keep_probs,
                                                 state_keep_prob=self._keep_probs,
                                                 variational_recurrent=True,
                                                 input_size=self._n_dim if i == 0 else self._layer_sizes_ed[i - 1],
                                                 dtype=tf.float32)
            cells.append(cell)
        # Multi cell layer
        cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        if state is None:
            output, state = tf.nn.dynamic_rnn(cells, inputs=inputs, dtype=tf.float32)
        else:
            output, state = tf.nn.dynamic_rnn(cells, inputs=inputs, initial_state=state, dtype=tf.float32)

        return output, state

    def _create_block_dense(self, inputs):
        net = inputs
        for i, units in enumerate(self._layer_sizes_f):
            net = tf.layers.dense(net, units=units, activation=self._activation, name='layer_%d' % i)
            net = tf.layers.dropout(net, rate=self._dropout)
        net = tf.layers.dense(net, units=1, activation=self._activation)
        output = tf.reshape(net, (-1, self._sle))
        return output
