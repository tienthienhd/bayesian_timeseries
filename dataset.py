import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def split_data(data, test_size=0.2):
    train, test = [], []
    n_trains = int(len(data[0]) * (1 - test_size))
    for col in data:
        train.append(col[:n_trains])
        test.append(col[n_trains:])
    return train, test


class GgTraceDataSet(object):

    def __init__(self, file_path, sliding_encoder, sliding_decoder):
        self.file_path = file_path
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder

        data_std = self.read_data()
        self.n_examples = data_std.shape[0]
        self.n_dim = data_std.shape[1]

        self.min_max_scaler = MinMaxScaler()
        self.data_scaled = self.min_max_scaler.fit_transform(data_std)

    def read_data(self):
        full_features = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
                         "meanCPUUsage", "canonical_memory_usage", "AssignMem",
                         "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
                         "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
                         "max_disk_io_time", "cpi", "mai", "sampling_portion",
                         "agg_type", "sampled_cpu_usage"]
        df = pd.read_csv(self.file_path, header=None, usecols=[3], names=full_features)
        # print(df.describe())
        return df

    def get_data(self, main_features=True):
        data_sup = series_to_supervised(self.data_scaled, n_in=self.sliding_encoder, n_out=self.sliding_decoder).values
        data_sup = data_sup.reshape((-1, self.sliding_encoder + self.sliding_decoder, self.n_dim))
        self.n_examples = data_sup.shape[0]

        start_input_dec = self.sliding_encoder - self.sliding_decoder

        inputs_enc = data_sup[:, :self.sliding_encoder]
        inputs_dec = data_sup[:, start_input_dec: self.sliding_encoder]

        outputs_dec = data_sup[:, self.sliding_encoder:]
        outputs = data_sup[:, self.sliding_encoder:]
        if main_features:
            outputs_dec = data_sup[:, self.sliding_encoder:, :1]
            outputs = data_sup[:, self.sliding_encoder:, 0]

        return inputs_enc, inputs_dec, outputs_dec, outputs

    def plot(self):
        self.data_scaled.plot()
        plt.show()

    def invert_transform(self, x, main_features=True):
        x = np.asarray(x)
        if main_features:
            min_feature = self.min_max_scaler.data_min_[0]
            max_feature = self.min_max_scaler.data_max_[0]
            return x * (max_feature - min_feature) + min_feature

        else:
            return self.min_max_scaler.inverse_transform(x)

# a = GgTraceDataSet('datasets/5.csv', 4, 2)
# b = a.get_data(main_features=True)
# c = split_data(b, test_size=0.6)
# print(len(c[0][0]))
