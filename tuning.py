import matplotlib.pyplot as plt
import numpy as np
from dataset import GgTraceDataSet2, split_data
from model import Model
import multiprocessing as mp
from sklearn.model_selection import ParameterGrid

dict_config = {
    "sliding_encoder": [8, 12, 16, 20, 24, 28, 32, 36, 40],
    "sliding_decoder": [1, 2, 3, 4, 5, 6, 7],
    "layer_sizes_ed": [[8], [16], [32], [64], [8, 4], [16, 8], [16, 4], [32, 16], [32, 4], [64, 32], [64, 16]],
    "layer_sizes_f": [[4], [8], [16], [32], [64], [8, 4], [16, 8], [16, 4], [32, 16], [32, 8], [64, 32], [64, 16], [64, 8]],
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

def run(params):
    model_type = 'f'
    # pprint.pprint(params)

    dataset = GgTraceDataSet2('datasets/5.csv', params['sliding_encoder'], params['sliding_decoder'])
    params['n_dim'] = dataset.n_dim
    if model_type == 'ed':
        data = dataset.get_data_ed()
        train, test = split_data(data, test_size=0.2)
        x_train = (train[0], train[1])
        y_train = train[2]
        x_test = (test[0], test[1])
        y_test = test[2]
    elif model_type == 'f':
        data = dataset.get_data_forecast()
        train, test = split_data(data, test_size=0.2)
        x_train = train[0]
        y_train = train[1]
        x_test = test[0]
        y_test = test[1]


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
    print('Running config: ' + model_name)

    model = Model('logs/' + model_name)
    model.build_model(params)
    # model.restore()
    history = model.train(x_train, y_train,
                          batch_size=params['batch_size'],
                          epochs=params['epochs'], verbose=1, model=model_type)
    # model.save()

    # plot history
    # plt.plot(history['loss'], label='loss')
    # plt.plot(history['val_loss'], label='val_loss')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()
    # plt.savefig('logs/' + model_name + '_history.png')
    # plt.clf()

    # plot predict
    preds = model.predict(x_test, model=model_type)
    preds_inv = dataset.invert_transform(preds)
    y_test_inv = dataset.invert_transform(y_test)

    mae = np.mean(np.abs(np.subtract(preds_inv, y_test_inv)))
    with open('logs/mae.csv', 'a') as f:
        f.write("{};{:.5f}\n".format(model_name, mae))

    # print(y_test_inv.shape, preds_inv.shape)
    if model_type == 'ed':
        y_test_inv = y_test_inv[:, -1, 0]
        preds_inv = preds_inv[:, -1, 0]
    elif model_type == 'f':
        y_test_inv = y_test_inv[:, 0]
        preds_inv = preds_inv[:, 0]

    plt.plot(y_test_inv, label='actual', color='#fc6b00', linestyle='solid')
    plt.plot(preds_inv, label='predict', color='blue', linestyle='solid')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.title('mae={:.2f}'.format(mae))
    plt.show()
    # plt.savefig('logs/' + model_name + '_predict.png')
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
    run(test_config)
else:
    mutil_running(list_configs=list_config, n_jobs=args.n_jobs)


