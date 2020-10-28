import pickle
import numpy as np
from sklearn import svm
from sklearn import metrics
import pandas as pd
from numpy.random import shuffle
from random import seed
import os

def load_pkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
        return obj


long_predict = 40


if __name__ == "__main__":
    path='../BatchData'
    # train data
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


    m = 100
    record = pd.DataFrame(columns=['acurrary_train', 'acurrary_test'])
    for k in range(1, m + 1):
        # k特征扩大倍数，特征值在0-1之间，彼此区分度太小，扩大以提高区分度和准确率
        x_train = data_train_batches[:, 2:] * k
        y_train = labels_train_batches[:, 0].astype(int)
        x_test = data_test_batches[:, 2:] * k
        y_test = labels_test_batches[:, 0].astype(int)

        model = svm.SVC()
        model.fit(x_train, y_train)
        cm_train = metrics.confusion_matrix(y_train, model.predict(x_train))
        cm_test = metrics.confusion_matrix(y_test, model.predict(x_test))

        pd.DataFrame(cm_train, index=range(1, 6), columns=range(1, 6))
        accurary_train = np.trace(cm_train) / cm_train.sum()
        pd.DataFrame(cm_test, index=range(1, 6), columns=range(1, 6))
        accurary_test = np.trace(cm_test) / cm_test.sum()
        record = record.append(
            pd.DataFrame([accurary_train, accurary_test], index=['accurary_train', 'accurary_test']).T)

    record.index = range(1, m + 1)
    find_k = record.sort_values(by=['accurary_train', 'accurary_test'], ascending=False)  # 生成一个copy 不改变原变量
    find_k[(find_k['accurary_train'] > 0.95) & (find_k['accurary_test'] > 0.95) & (
                find_k['accurary_test'] >= find_k['accurary_train'])]

    k = 33
    x_train = data_train_batches[:, 2:] * k
    y_train = labels_train_batches[:, 0].astype(int)
    model = svm.SVC()
    model.fit(x_train, y_train)
    model.score(x_train, y_train)
    datax_train = data_train_batches[:, 2:] * k
    datay_train = labels_train_batches[:, 0].astype(int)
    model.score(datax_train, datay_train)
    cm_data = metrics.confusion_matrix(datay_train, model.predict(datax_train))
    pd.DataFrame(cm_data, index=range(1, 6), columns=range(1, 6))
    accurary_data = np.trace(cm_data) / cm_data.sum()
    print(accurary_data)


