from tensorflow.python.keras.models import Model, model_from_json, load_model
from tensorflow.python.keras.layers import LSTM, Input, Dense, GRU, Lambda
from tensorflow.python.keras.callbacks import EarlyStopping, \
    ReduceLROnPlateau, TerminateOnNaN, TensorBoard, LearningRateScheduler
# from tensorflow.python.keras.utils import plot_model
import numpy as np
import json


# import pprint


class EDModel(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.ed_model = None
        self.encoder_model = None
        self.params = None

    def _parse_params(self, params):
        self.params = params
        self.sliding_encoder = params['sliding_encoder']
        self.sliding_decoder = params['sliding_decoder']
        self.layer_sizes_ed = params['layer_sizes_ed']
        # self.layer_sizes_ann = params['layer_sizes_ann']
        self.n_dim = params['n_dim']
        self.activation = params['activation']
        self.optimizer = params['optimizer']
        self.learning_rate = params['learning_rate']
        self.dropout = params['dropout']
        self.recurrent_dropout = params['recurrent_dropout']
        cell_type = params['cell_type']
        if cell_type == 'lstm':
            self.cell = LSTM
        elif cell_type == 'gru':
            self.cell = GRU

    def build_model(self, params):
        self._parse_params(params)
        # ========================ENCODER=====================
        en_state = []
        en_output = None
        en_input = Input(shape=(self.sliding_encoder, self.n_dim), name='encoder_input')

        for i, units in enumerate(self.layer_sizes_ed):
            encoder_layer = self.cell(units=units,
                                      activation=self.activation,
                                      recurrent_dropout=self.recurrent_dropout,
                                      dropout=self.dropout,
                                      return_sequences=True,
                                      return_state=True,
                                      name='encoder_layer_' + str(i))
            if i == 0:
                en_output, h, c = encoder_layer(en_input)
            else:
                en_output, h, c = encoder_layer(en_output)
            en_state.append([h, c])
        feature = en_output

        # ========================DECODER=====================
        de_output = None
        de_input = Input(shape=(self.sliding_decoder, self.n_dim), name='decoder_input')
        for i, units in enumerate(self.layer_sizes_ed):
            decoder_layer = self.cell(units=units,
                                      activation=self.activation,
                                      recurrent_dropout=self.recurrent_dropout,
                                      dropout=self.dropout,
                                      return_sequences=True,
                                      return_state=True,
                                      name='decoder_layer_' + str(i))
            if i == 0:
                de_output, h, c = decoder_layer(de_input, initial_state=en_state[i])
            else:
                de_output, h, c = decoder_layer(de_output, initial_state=en_state[i])

        ed_preds = Dense(units=1, name='ed_preds')(de_output)

        # ========================= Model ED ========================
        ed_model = Model([en_input, de_input], ed_preds)
        ed_model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])
        # ed_model.summary(line_length=200)
        self.ed_model = ed_model

        # ========================= ENCODER MODEL ===================
        self.encoder_model = Model(en_input, feature)

    def train(self, x, y, batch_size, epochs, verbose):
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
        history = self.ed_model.fit(x=x, y=y,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    callbacks=callbacks,
                                    shuffle=False,
                                    validation_split=0.2,
                                    verbose=verbose)
        return history.history

    def save(self):

        self.ed_model.save(self.model_dir + '/ed_model.h5')
        self.encoder_model.save(self.model_dir + '/encoder_model.h5')

        with open(self.model_dir + "/params.json", 'w') as f:
            json.dump(self.params, f)

    def restore(self):
        print("Loading model....")
        self.ed_model = load_model(self.model_dir + '/ed_model.h5')
        self.encoder_model = load_model(self.model_dir + '/encoder_model.h5')

        self.params = json.load(open(self.model_dir + '/params.json', 'r'))
        self.params = self.params
        self.sliding_encoder = self.params['sliding_encoder']
        self.sliding_decoder = self.params['sliding_decoder']
        self.layer_sizes_ed = self.params['layer_sizes_ed']
        # self.layer_sizes_ann = self.params['layer_sizes_ann']
        self.n_dim = self.params['n_dim']
        self.activation = self.params['activation']
        self.optimizer = self.params['optimizer']
        self.learning_rate = self.params['learning_rate']
        self.dropout = self.params['dropout']
        self.recurrent_dropout = self.params['recurrent_dropout']
        cell_type = self.params['cell_type']
        if cell_type == 'lstm':
            self.cell = LSTM
        elif cell_type == 'gru':
            self.cell = GRU

    def eval(self, x, y):
        return self.ed_model.evaluate(x, y)

    def predict(self, x):
        return self.ed_model.predict(x)

    def get_features(self, x):
        return self.encoder_model.predict(x)
