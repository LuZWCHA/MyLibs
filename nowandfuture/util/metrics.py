import math

import numpy as np
from sklearn import metrics


# input should be format of CxWxHxL or CxWxH
def dice(g: np.ndarray, label: np.ndarray):
    """

    :param g: generate img format of numpy array
    :param label: the label
    :return:
    """
    if len(g.shape) < 3:
        g = g[np.newaxis, ...]
    if len(label.shape) < 3:
        label = label[np.newaxis, ...]

    if len(g.shape) != len(label.shape):
        raise RuntimeError('the lengths of the inputs must be equal')

    dims = tuple([x for x in range(1, len(g.shape))])
    num = g * label
    num_c = np.sum(num, axis=dims)
    den1 = g * g
    den1_c = np.sum(den1, axis=dims)
    den2 = label * label
    den2_c = np.sum(den2, axis=dims)

    dice_all_channel = 2 * ((num_c + 0.0000001) / (den1_c + den2_c + 0.0000001))
    channel_size = dice_all_channel.shape[0]

    dice_total = np.sum(dice_all_channel) / channel_size

    return dice_total


def RMSE(g: np.ndarray, label: np.ndarray):
    error = g - label
    mse = np.average(error * error)
    return math.sqrt(mse)


def ncc(a: np.ndarray, b: np.ndarray):
    a = (a - a.mean())
    b = (b - b.mean())
    c = np.multiply(a, b)
    c = c / (a.std() * b.std())
    return c


def MI(labels_true: np.ndarray, labels_pre: np.ndarray):
    labels_true = labels_true.flatten()
    labels_pre = labels_pre.flatten()
    return metrics.mutual_info_score(labels_true, labels_pre)


def NMI(labels_true: np.ndarray, labels_pre: np.ndarray):
    labels_true = labels_true.flatten()
    labels_pre = labels_pre.flatten()
    return metrics.normalized_mutual_info_score(labels_true, labels_pre)
