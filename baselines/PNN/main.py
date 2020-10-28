import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

import config
from PNN import PNN


def load_pkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
        return obj


def run_base_model_pnn(train_x, tarin_y,test_x,text_y,pnn_params):
    for i in range(number_train_batches):
        batch_x, batch_y = train_x[i], tarin_y[i]
        batch_x_, batch_y_ = train_x[i], tarin_y[i]

        pnn = PNN(**pnn_params)
        pnn.fit(i, batch_x, batch_y, i, batch_x_, batch_y_)



if __name__ == "__main__":

    pnn_params = {
        "embedding_size": 64,
        "deep_layers": [32, 32],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layer_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "verbose": True,
        "random_seed": config.RANDOM_SEED,
        "deep_init_size": 50,
        "use_inner": False

    }
    # load data
    path = '../BatchData'
    dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()
    path_string = path + '/TrainData.seqs'
    data_train_batches = load_pkl(path_string)

    path_string = path + '/TrainLabel.seqs'
    labels_train_batches = load_pkl(path_string)

    number_train_batches = len(data_train_batches)

    input_dim = np.array(data_train_batches[0]).shape[2]
    output_dim = np.array(labels_train_batches[0]).shape[1]

    print("Train data is loaded!")

    path_string = path + '/TestData.seqs'
    data_test_batches = load_pkl(path_string)

    path_string = path + '/TestLabel.seqs'
    labels_test_batches = load_pkl(path_string)

    number_test_batches = len(data_test_batches)

    print("Test data is loaded!")

    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))

    y_train_pnn, y_test_pnn = run_base_model_pnn(data_test_batches, labels_test_batches, folds, pnn_params)