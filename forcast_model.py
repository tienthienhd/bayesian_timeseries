import json

from tensorflow.python.keras.models import Model, model_from_json, load_model
from tensorflow.python.keras.layers import LSTM, Input, Dense, GRU, Lambda, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, \
    ReduceLROnPlateau, TerminateOnNaN, TensorBoard, LearningRateScheduler
# from tensorflow.python.keras.utils import plot_model
import numpy as np
# import json
# import pprint
from ed_model import EDModel


class AnnModel(object):
    def __init__(self, model_dir, model_ed_dir):
        self.model_dir = model_dir
        self.model = None
        self.ed_model = EDModel(model_ed_dir)
        self.ed_model.restore()

    def _parse_params(self, params):
        self.params = params
        self.layer_sizes_ann = params['layer_sizes_ann']
        self.n_dim = params['n_dim']
        self.activation = params['activation']
        self.optimizer = params['optimizer']
        self.learning_rate = params['learning_rate']
        self.dropout = params['dropout']

    def build_model(self, params):
        self._parse_params(params)

        inputs = Input(shape=(18,), name='input_ann')

        net = None
        for i, units in enumerate(self.layer_sizes_ann):
            layer = Dense(units=units, activation=self.activation, name='layer_' + str(i))
            dropout = Dropout(rate=self.dropout)
            if i == 0:
                net = layer(inputs)
            else:
                net = layer(net)
            net = dropout(net, training=True)

        preds = Dense(units=1, name='output')(net)

        model = Model(inputs, preds)
        model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])
        model.summary(line_length=200)
        self.model = model

    def _compute_features(self, x):
        features = self.ed_model.get_features(x)
        features = np.mean(features, axis=2)
        # print('features shape:', features.shape)
        return features

    def train(self, x, y, batch_size, epochs, verbose):
        features = self._compute_features(x)

        def exp_decay(epoch):
            initial_lrate = self.learning_rate
            k = 0.1
            lrate = initial_lrate * np.exp(-k * epoch)
            return lrate

        callbacks = [
            EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True),
            LearningRateScheduler(exp_decay, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0),
            TensorBoard(log_dir=self.model_dir),
            TerminateOnNaN()
        ]

        self.model.fit(x=features, y=y,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=callbacks,
                       shuffle=False,
                       validation_split=0.2,
                       verbose=verbose)

    def eval(self, x, y):
        features = self._compute_features(x)
        return self.model.evaluate(features, y)

    def predict(self, x):
        features = self._compute_features(x)
        return self.model.predict(features)

    def save(self):
        self.model.save(self.model_dir + '/ann_model.h5')
        json.dump(self.params, open(self.model_dir+'/params.json', 'w'))

    def restore(self):
        print("Loading model....")
        self.model = load_model(self.model_dir + '/ann_model.h5')
        self.params = json.load(open(self.model_dir + '/params.json', 'r'))
        self.layer_sizes_ann = self.params['layer_sizes_ann']
        self.n_dim = self.params['n_dim']
        self.activation = self.params['activation']
        self.optimizer = self.params['optimizer']
        self.learning_rate = self.params['learning_rate']
        self.dropout = self.params['dropout']


#
from dataset import GgTraceDataSet, split_data


params = {
    'sliding_encoder': 18,
    'sliding_decoder': 6,
    'layer_sizes_ed': [16, 8],
    'layer_sizes_ann': [32, 4],
    'activation': 'tanh',
    'optimizer': 'rmsprop',
    # 'n_dim': 1,
    'dropout': 0.5,
    'recurrent_dropout': 0.1,
    'batch_size': 32,
    'learning_rate': 0.01,
    'epochs': 200,
    'cell_type': 'lstm',
}

dataset = GgTraceDataSet('datasets/5.csv', params['sliding_encoder'], params['sliding_decoder'])
params['n_dim'] = dataset.n_dim
data = dataset.get_data()
train, test = split_data(data, test_size=0.2)
x_train = train[0]
y_train = train[3][:, 0].reshape((-1, 1))
x_test = test[0]
y_test = test[3][:, 0].reshape((-1, 1))

print(y_train.shape)

model = AnnModel('logs/test_ann', 'logs/test2')
# model.build_model(params)
# model.train(x_train, y_train,
#             batch_size=params['batch_size'],
#             epochs=params['epochs'],
#             verbose=1)

# model.save()
model.restore()
print(model.params)
# print(model.eval(x_test, y_test))
# print(model.eval(x_test, y_test))
# preds = model.predict(x_test)
#
#
# import matplotlib.pyplot as plt
#
# plt.plot(y_test)
# plt.plot(preds)
# plt.legend(['actual', 'predict'])
#
# plt.show()