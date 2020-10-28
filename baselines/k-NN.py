import pickle
import numpy as np
from sklearn import neighbors

def load_pkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
        return obj

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

    knnModel = neighbors.KNeighborsClassifier(n_neighbors=3)


    # 用fit方法训练模型，传入特征集和训练集的数据
    knnModel.fit(data_train_batches, data_train_batches)

    # test
    knnModel.score(data_train_batches, data_train_batches)

    from sklearn.model_selection import cross_val_score

    cross_val_score(
        knnModel,
        data_train_batches, data_train_batches, cv=5  # K折交叉验证的值，因为需要5折所以输入5
    )

    # 使用模型进行预测
    knnModel.predict(data_test_batches)



