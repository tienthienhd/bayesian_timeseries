import time

import tensorflow as tf
import numpy as np
import json
import utils



class Model(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.layer_sizes_ed = []
        self.layer_sizes_f = []
        self.activation = tf.nn.tanh
        self.keep_probs = 0.95
        self.dropout = 0.05
        self.sle = 1
        self.sld = 1
        self.n_dim = 1
        self.optimizer = tf.train.AdamOptimizer
        self.learning_rate = 0.1
        self.cell_type = tf.nn.rnn_cell.LSTMCell
        self.patience = 1


        self.xe = None
        self.xd = None
        self.yd = None
        self.yf = None
        self.pred_d = None
        self.pred_f = None
        self.loss_ed = None
        self.loss_f = None
        self.mae_ed = None
        self.mae_f = None
        self.rmse_ed = None
        self.rmse_f = None
        self.train_op_ed = None
        self.train_op_f = None

        tf.reset_default_graph()
        self.sess = tf.Session()

    def build_model(self, params):
        self._parse_params(params)

        # input
        self.xe = tf.placeholder(tf.float32, (None, self.sle, self.n_dim), 'ex')
        self.xd = tf.placeholder(tf.float32, (None, self.sld, self.n_dim), 'dx')
        self.yd = tf.placeholder(tf.float32, (None, self.sld, 1), 'dy')
        self.yf = tf.placeholder(tf.float32, (None, 1), 'fy')

        # encoder
        with tf.variable_scope('encoder'):
            out_e, state_e = self._create_block_rnn(self.xe, state=None)

        # decoder
        with tf.variable_scope('decoder'):
            out_d, state_d = self._create_block_rnn(self.xd, state=state_e)
            self.pred_d = tf.layers.dense(out_d, units=1)

        # forecaster
        with tf.variable_scope('forecaster'):
            out_e = tf.stop_gradient(out_e, 'features')
            out_f = self._create_block_dense(out_e)
            self.pred_f = tf.layers.dense(out_f, units=1)

        # loss
        with tf.variable_scope('loss'):
            self.loss_ed = tf.losses.mean_squared_error(self.yd, self.pred_d)
            self.loss_f = tf.losses.mean_squared_error(self.yf, self.pred_f)

        # metrics
        with tf.variable_scope('metrics'):
            self.mae_ed = tf.reduce_mean(tf.abs(tf.subtract(self.yd, self.pred_d)))
            self.rmse_ed = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.yd, self.pred_d))))

            self.mae_f = tf.reduce_mean(tf.abs(tf.subtract(self.yf, self.pred_f)))
            self.rmse_f = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.yf, self.pred_f))))

        # optimizer
        with tf.variable_scope('optimizer'):
            self.train_op_ed = self.optimizer(self.learning_rate).minimize(self.loss_ed)
            self.train_op_f = self.optimizer(self.learning_rate).minimize(self.loss_f)

        # add to collection
        tf.add_to_collection('params', self.xe)
        tf.add_to_collection('params', self.xd)
        tf.add_to_collection('params', self.yd)
        tf.add_to_collection('params', self.yf)
        tf.add_to_collection('params', self.pred_d)
        tf.add_to_collection('params', self.pred_f)
        tf.add_to_collection('params', self.loss_ed)
        tf.add_to_collection('params', self.loss_f)
        tf.add_to_collection('params', self.mae_ed)
        tf.add_to_collection('params', self.mae_f)
        tf.add_to_collection('params', self.rmse_ed)
        tf.add_to_collection('params', self.rmse_f)
        tf.add_to_collection('params', self.train_op_ed)
        tf.add_to_collection('params', self.train_op_f)

        # initial variable
        self.sess.run(tf.global_variables_initializer())

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.model_dir + '/model')

        if hasattr(self, 'params'):
            with open(self.model_dir + "/hyper_params.json", 'w') as f:
                json.dump(self.params, f)

    def restore(self):
        saver = tf.train.import_meta_graph(self.model_dir + '/model.meta')
        saver.restore(self.sess, self.model_dir + '/model')
        params = tf.get_collection('params')
        self.xe = params[0]
        self.xd = params[1]
        self.yd = params[2]
        self.yf = params[3]
        self.pred_d = params[4]
        self.pred_f = params[5]
        self.loss_ed = params[6]
        self.loss_f = params[7]
        self.mae_ed = params[8]
        self.mae_f = params[9]
        self.rmse_ed = params[10]
        self.rmse_f = params[11]
        self.train_op_ed = params[12]
        self.train_op_f = params[13]

    def train(self, x, y, validation_split=0.2, batch_size=32, epochs=1, verbose=1, model='ed'):

        if model == 'ed':
            return self._train_ed(x, y, validation_split, batch_size, epochs, verbose)
        elif model == 'f':
            return self._train_f(x, y, validation_split, batch_size, epochs, verbose)

    def eval(self, x, y, model='ed'):
        if model == 'ed':
            return self.sess.run([self.loss_ed, self.mae_ed], {
                self.xe: x[0],
                self.xd: x[1],
                self.yd: y
            })
        elif model == 'f':
            return self.sess.run([self.loss_f, self.mae_f], {
                self.xe: x,
                self.yf: y
            })

    def predict(self, x, model='ed'):
        if model == 'ed':
            return self.sess.run(self.pred_d, {self.xe: x[0], self.xd: x[1]})
        elif model == 'f':
            return self.sess.run(self.pred_f, {self.xe: x})

    def _parse_params(self, params):
        self.params = params
        self.sle = int(params['sliding_encoder'])
        self.sld = int(params['sliding_decoder'])
        self.layer_sizes_ed = params['layer_sizes_ed']
        self.layer_sizes_f = params['layer_sizes_f']
        self.n_dim = int(params['n_dim'])
        self.keep_probs = params['keep_probs']
        self.dropout = params['dropout']
        self.patience = params['patience']
        self.learning_rate = params['learning_rate']

        self.activation = utils.transform_activation(params['activation'])
        self.optimizer = utils.transform_optimizer(params['optimizer'])
        self.cell_type = utils.transform_cell_type(params['cell_type'])


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

    def _create_block_dense(self, inputs):
        net = inputs
        for i, units in enumerate(self.layer_sizes_f):
            net = tf.layers.dense(net, units=units, activation=self.activation, name='layer_%d' % i)
            net = tf.layers.dropout(net, rate=self.dropout)
        net = tf.layers.dense(net, units=1, activation=self.activation)
        output = tf.reshape(net, (-1, self.sle))
        return output

    def _train_ed(self, x, y, validation_split, batch_size, epochs, verbose):
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
                    self.xe: xe,
                    self.xd: xd,
                    self.yd: yd
                }
                output_feed = [self.loss_ed, self.mae_ed, self.train_op_ed]

                try:
                    l, m, _ = self.sess.run(output_feed, input_feed)

                    loss += l
                    mae += m
                except ValueError as e:
                    print('==========> Exception: {}'.format(str(self.params)))
                    print('________{}'.format(xd.shape))
                    print(e)
                    break
            loss /= n_batches
            mae /= n_batches
            history['loss'].append(loss)
            history['mae'].append(mae)

            try:
                val_loss, val_mae = self.sess.run([self.loss_ed, self.mae_ed],
                                                  {
                                                      self.xe: xe_val,
                                                      self.xd: xd_val,
                                                      self.yd: yd_val
                                                  })
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
            except ValueError as e:
                print('==========> Exception: {}'.format(str(self.params)))
                print('________{}'.format(xd_val.shape))
                print(e)
                break

            epoch_time = time.time() - start_epoch_time
            if verbose > 0:
                print(
                    "Epoch {}/{}: time={:.2f}s, loss={:.5f}, mae={:.5f}, val_loss={:.5f}, val_mae={:.5f}".format(
                        e + 1, epochs, epoch_time,
                        loss, mae, val_loss, val_mae))
            if utils.early_stop(history['val_loss'], e, patience=self.patience):
                print('Early stop at epoch', (e + 1))
                break
            if np.isnan(loss):
                break
        return history

    def _train_f(self, x, y, validation_split, batch_size, epochs, verbose):
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
                    self.xe: xe,
                    self.yf: yf
                }
                output_feed = [self.loss_f, self.mae_f, self.train_op_f]
                l, m, _ = self.sess.run(output_feed, input_feed)

                loss += l
                mae += m
            loss /= n_batches
            mae /= n_batches
            history['loss'].append(loss)
            history['mae'].append(mae)

            val_loss, val_mae = self.sess.run([self.loss_f, self.mae_f],
                                              {
                                                  self.xe: xe_val,
                                                  self.yf: yf_val
                                              })
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            epoch_time = time.time() - start_epoch_time
            if verbose > 0:
                print(
                    "Epoch {}/{}: time={:.2f}s, loss={:.5f}, mae={:.5f}, val_loss={:.5f}, val_mae={:.5f}".format(
                        e + 1, epochs, epoch_time,
                        loss, mae, val_loss, val_mae))
            if utils.early_stop(history['val_loss'], e, patience=self.patience):
                print('Early stop at epoch', (e + 1))
                break
            if np.isnan(loss):
                break
        return history