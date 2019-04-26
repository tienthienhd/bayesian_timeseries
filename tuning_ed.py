import random
import matplotlib.pyplot as plt
import numpy as np
from dataset import GgTraceDataSet, split_data
from ed_model import EDModel
import multiprocessing as mp
from sklearn.model_selection import ParameterGrid, ParameterSampler

dict_config = {
    "sliding_encoder": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    "sliding_decoder": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "layer_sizes_ed": [[8], [16], [32], [64], [8, 4], [16, 8], [16, 4], [32, 16], [32, 4], [64, 32], [64, 16]],
    "activation": ['tanh', 'sigmoid'],
    "optimizer": ['adam', 'rmsprop'],
    "batch_size": [8, 16, 32, 64],
    "cell_type": ['lstm'],
    "epochs": [2],
    "keep_probs": [0.95],
    "learning_rate": [0.0001, 0.001, 0.01],
    "patience": [15],
}

def run(params):
    # pprint.pprint(params)

    dataset = GgTraceDataSet('datasets/5.csv', params['sliding_encoder'], params['sliding_decoder'])
    params['n_dim'] = dataset.n_dim
    data = dataset.get_data()
    train, test = split_data(data, test_size=0.2)
    x_train = (train[0], train[1])
    y_train = train[2]
    x_test = (test[0], test[1])
    y_test = test[2]

    model_name = "sle({})_sld({})_ls({})_ac({})_opt({})_kp({})_bs({})_lr({})_ct({})".format(
        params['sliding_encoder'],
        params['sliding_decoder'],
        params['layer_sizes_ed'],
        params['activation'],
        params['optimizer'],
        params['keep_probs'],
        params['batch_size'],
        params['learning_rate'],
        params['cell_type']
    )
    print('Running config: ' + model_name)

    model = EDModel('logs/' + model_name)
    model.build_model(params)
    history = model.train(x_train, y_train,
                          batch_size=params['batch_size'],
                          epochs=params['epochs'], verbose=1)
    # model.save()

    # plot history
    # histor = pd.DataFrame(history)
    # print(histor.describe())
    # history.loc[:, ['loss', 'val_loss']].plot()
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.show()
    plt.savefig('logs/' + model_name + '_history.png')
    plt.clf()

    # plot predict
    preds = model.predict(x_test)
    mae = np.mean(np.abs(y_test - preds)) * 56.863121

    with open('logs/mae.csv', 'a') as f:
        f.write("{};{:.5f}\n".format(model_name, mae))

    y_test = y_test[:, -1, 0]
    preds = preds[:, -1, 0]
    plt.plot(y_test, label='actual')
    plt.plot(preds, label='predict')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('logs/' + model_name + '_predict.png')
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
        pool.map(run, list_configs_map)

    pool.close()
    pool.join()
    pool.terminate()


test_config = {
    'sliding_encoder': 30,
    'sliding_decoder': 6,
    'layer_sizes_ed': [64, 16],
    'activation': 'tanh',
    'optimizer': 'rmsprop',
    'input_keep_prob': 0.95,
    'output_keep_prob': 0.95,
    'state_keep_prob': 0.95,
    'batch_size': 2,
    'learning_rate': 0.001,
    'epochs': 200,
    'cell_type': 'lstm',
    'patience': 2
}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', default=False, type=bool)
parser.add_argument('--n_jobs', default=1, type=int)
parser.add_argument('--n_configs', default=1, type=int)
args = parser.parse_args()

list_config = np.random.choice(list(ParameterGrid(dict_config)), size=args.n_configs)
print(len(list_config))

if args.test:
    run(test_config)
else:
    mutil_running(list_configs=list_config, n_jobs=args.n_jobs)
