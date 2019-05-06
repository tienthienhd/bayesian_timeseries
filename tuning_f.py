import matplotlib.pyplot as plt
import numpy as np
from dataset import GgTraceDataSet2, split_data
from model import Model
import multiprocessing as mp
from sklearn.model_selection import ParameterGrid

dict_config = {
    "sliding_encoder": [24, 32],
    "sliding_decoder": [1, 2],
    "layer_sizes_ed": [[8], [64], [16, 8], [32, 16]],
    "layer_sizes_f": [[4], [8], [16], [32], [64], [8, 4], [16, 8], [16, 4], [32, 16], [32, 8], [64, 32], [64, 16],
                      [64, 8]],
    "activation": ['tanh', 'sigmoid'],
    "optimizer": ['adam', 'rmsprop'],
    "batch_size": [8, 16, 32, 64],
    "cell_type": ['lstm'],
    "epochs": [500],
    "keep_probs": [0.95],
    "dropout": [0.05],
    "learning_rate": [0.0001, 0.001, 0.01],
    "patience": [15],
}

def run_ed(params):
    dataset = GgTraceDataSet2('datasets/5.csv', params['sliding_encoder'], params['sliding_decoder'])
    params['n_dim'] = dataset.n_dim

    data_ed = dataset.get_data_ed()
    train_ed, test_ed = split_data(data_ed, test_size=0.2)
    x_train_ed = train_ed[:2]
    y_train_ed = train_ed[-1]

    x_test_ed = test_ed[:2]
    y_test_ed = test_ed[-1]

    data_f = dataset.get_data_forecast()
    train_f, test_f = split_data(data_f, test_size=0.2)
    x_train_f = train_f[0]
    y_train_f = train_f[-1]

    x_test_f = test_f[0]
    y_test_f = test_f[-1]

    model_name = "sle({})_sld({})_lsed({})_lsf({})_ac({})_opt({})_kp({})_drop({})_bs({})_lr({})_ct({})_pat({})".format(
        params['sliding_encoder'],
        params['sliding_decoder'],
        params['layer_sizes_ed'],
        params['layer_sizes_f'],
        params['activation'],
        params['optimizer'],
        params['keep_probs'],
        params['dropout'],
        params['batch_size'],
        params['learning_rate'],
        params['cell_type'],
        params['patience']
    )

    print('Runing config:' + model_name)

    model = Model('logs/' + model_name, params=params)

    model.train_ed(x_train_ed, y_train_ed, batch_size=params['batch_size'], epochs=params['epochs'], verbose=1)

    model.train_f(x_train_f, y_train_f, batch_size=params['batch_size'], epochs=params['epochs'], verbose=1)

    preds = model.predict_f(x_test_f)
    preds_inv = dataset.invert_transform(preds)
    y_test_inv = dataset.invert_transform(y_test_f)

    mae = np.mean(np.abs(np.subtract(preds_inv, y_test_inv)))
    with open('logs/mae.csv', 'a') as f:
        f.write("{};{:.5f}\n".format(model_name, mae))

    preds_inv = preds_inv[:, 0]
    y_test_inv = y_test_inv[:, 0]

    plt.plot(y_test_inv, label='actual', color='#fc6b00', linestyle='solid')
    plt.plot(preds_inv, label='predict', color='blue', linestyle='solid')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.title('mae={:.2f}'.format(mae))
    plt.show()
    # plt.savefig('logs/' + str(mae) + "_" + model_name + '_predict_f.png')
    plt.clf()


def mutil_running(list_configs, n_jobs=1):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    pool = mp.Pool(n_jobs)

    num_configs = len(list_configs)
    config_per_map = 64
    n_maps = num_configs // config_per_map
    if num_configs % config_per_map != 0:
        n_maps += 1

    for i in range(n_maps):
        list_configs_map = list_configs[i * config_per_map: (i + 1) * config_per_map]
        pool.map(run_ed, list_configs_map)

    pool.close()
    pool.join()
    pool.terminate()


test_config = {
    'sliding_encoder': 24,
    'sliding_decoder': 1,
    'layer_sizes_ed': [64],
    'layer_sizes_f': [64, 16],
    'activation': 'tanh',
    'optimizer': 'rmsprop',
    'keep_probs': 0.95,
    'dropout': 0.05,
    'batch_size': 32,
    'learning_rate': 0.01,
    'epochs': 500,
    'cell_type': 'lstm',
    'patience': 15
}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', default=False, type=bool, choices=[True, False])
parser.add_argument('--n_jobs', default=1, type=int)
parser.add_argument('--n_configs', default=1, type=int)
args = parser.parse_args()

list_config = np.random.choice(list(ParameterGrid(dict_config)), size=args.n_configs)
print(len(list_config))

if args.test:
    run_ed(test_config)
else:
    mutil_running(list_configs=list_config, n_jobs=args.n_jobs)