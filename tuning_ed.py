import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset import GgTraceDataSet, split_data
from ed_model import EDModel
import pprint
import multiprocessing as mp

sliding_encoder = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
sliding_decoder = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# sliding_encoder = list(range(4, 100, 2))
# sliding_decoder = list(range(2, 50, 2))
layer_unit = list(range(0, 128))
num_layer = [1, 2, 3]
activation = ['tanh', 'sigmoid']
optimizer = ['adam', 'rmsprop']
dropout = [0.05, 0.8]
recurrent_dropout = [0.05, 0.8]
batch_size = [8, 16, 32, 64, 128]
learning_rate = [0.000001, 1]
cell_type = ['lstm']
epochs = 500


def next_config():
    sl_e = 0
    sl_d = 0
    while (True):
        sl_e = random.choice(sliding_encoder)
        sl_d = random.choice(sliding_decoder)
        if sl_e > sl_d:
            break

    # Layer size
    n_layer = random.choice(num_layer)
    layer_sizes = []
    for i in range(n_layer):
        if i == 0:
            while (True):
                units = random.choice(layer_unit)
                if units != 0:
                    layer_sizes.append(units)
                    break
        else:
            if layer_sizes[i - 1] == layer_unit[-1]:
                break

            while (True):
                units = random.choice(layer_unit)
                if 0 < units <= layer_sizes[i - 1]:
                    layer_sizes.append(units)
                    break

    # Activation
    ac = random.choice(activation)
    op = random.choice(optimizer)
    drop = random.uniform(dropout[0], dropout[1])
    rdrop = random.uniform(recurrent_dropout[0], recurrent_dropout[1])
    lr = random.uniform(learning_rate[0], learning_rate[1])
    bs = random.choice(batch_size)
    ct = random.choice(cell_type)

    config = {
        'sliding_encoder': sl_e,
        'sliding_decoder': sl_d,
        'layer_sizes_ed': layer_sizes,
        'activation': ac,
        'optimizer': op,
        'dropout': drop,
        'recurrent_dropout': rdrop,
        'batch_size': bs,
        'learning_rate': lr,
        'epochs': epochs,
        'cell_type': ct,
    }
    return config


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

    model_name = "sle({})_sld({})_ls({})_ac({})_opt({})_drop({})_rdrop({})_bs({})_lr({})_ct({})".format(
        params['sliding_encoder'],
        params['sliding_decoder'],
        params['layer_sizes_ed'],
        params['activation'],
        params['optimizer'],
        params['dropout'],
        params['recurrent_dropout'],
        params['batch_size'],
        params['learning_rate'],
        params['cell_type']
    )

    model = EDModel('logs/' + model_name)
    model.build_model(params)
    history = model.train(x_train, y_train,
                          params['batch_size'],
                          params['epochs'], verbose=0)

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
    plt.savefig('logs/' + model_name + '/history.png')
    plt.clf()

    # plot predict
    preds = model.predict(x_test)
    mae = np.mean(np.abs(y_test - preds)) * 56.863121

    with open('logs/mae.txt', 'a') as f:
        f.write("{},{:.5f}\n".format(model_name, mae))

    y_test = y_test[:, -1, 0]
    preds = preds[:, -1, 0]
    plt.plot(y_test, label='actual')
    plt.plot(preds, label='predict')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('logs/' + model_name + '/predict.png')
    plt.clf()

    model.save()


def mutil_running(num_configs=1, n_jobs=1):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    pool = mp.Pool(n_jobs)

    list_configs = [next_config() for i in range(num_configs)]
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


mutil_running(num_configs=200, n_jobs=8)

