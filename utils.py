import numpy as np
import torch
import random

"""
1.指标构建
因为实际任务中为多分类，选取macro_average作为处理方式.
"""


def _precision(y_pred, y_true, x):
    # 函数名这么取是为了区分于 get_xxxxxxx，否则自动补全时一次跳出七个待选函数
    correct = y_pred[y_pred == y_true]
    return sum(correct == x) / sum(y_pred == x)


def _recall(y_pred, y_true, x):
    correct = y_pred[y_pred == y_true]
    return sum(correct == x) / sum(y_true == x)


def _f1(y_pred, y_true, x):
    p = _precision(y_pred, y_true, x)
    r = _recall(y_pred, y_true, x)
    return 2 * p * r / (p + r)


def get_accuracy(y_pred, y_true):
    assert len(y_pred) == len(y_true), "Two lengths mismatch！"
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    right = y_pred == y_true
    return sum(right) / len(y_pred)


def get_macro_precision(y_pred, y_true):
    assert len(y_pred) == len(y_true), "Two lengths mismatch！"
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    precisions = []
    for x in np.unique(np.append(y_pred, y_true)):
        precisions.append(_precision(y_pred, y_true, x))
    return sum(precisions) / len(precisions)


def get_macro_recall(y_pred, y_true):
    assert len(y_pred) == len(y_true), "Two lengths mismatch！"
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    recalls = []
    for x in np.unique(np.append(y_pred, y_true)):
        recalls.append(_recall(y_pred, y_true, x))
    return sum(recalls) / len(recalls)


def get_macro_f1(y_pred, y_true):
    assert len(y_pred) == len(y_true), "Two lengths mismatch！"
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    f1s = []
    for x in np.unique(np.append(y_pred, y_true)):
        f1s.append(_recall(y_pred, y_true, x))
    return sum(f1s) / len(f1s)


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return torch.tensor(images).float(), torch.tensor(labels)


def data_iter(batch_size, X_train, y_train):
    # Batch data for training
    indices = list(range(len(X_train)))
    random.shuffle(indices)
    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i: min(i + batch_size, len(X_train))]
        yield X_train[batch_indices], y_train[batch_indices]
