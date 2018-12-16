from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt

import data_util

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp() + 1e-5).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        # import pdb; pdb.set_trace()
        return loss.sum(dim=1).mean()


def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()


def F1_soft(preds, targs, th=0.5, d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0 * (preds * targs).sum(axis=0)
    score /= (preds + targs).sum(axis=0) + 1e-6
    return score


def fit_val(x,y):
    params = 0.5 * np.ones(len(data_util.name_label_dict))
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x, y, p) - 1.0,
                                      wd * (p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def Count_soft(preds, th=0.5, d=50.0):
    preds = sigmoid_np(d * (preds - th))
    return preds.mean(axis=0)


def fit_test(x, y):
    params = 0.5*np.ones(len(data_util.name_label_dict))
    wd = 1e-5
    error = lambda p: np.concatenate((Count_soft(x,p) - y,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p
