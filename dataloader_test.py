from fastai.conv_learner import *
from fastai.dataset import *
import pretrainedmodels

import os
import warnings
import glob

import fire
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt

import ml_util
import torch_model
import data_util
import kernel

args = kernel.dataset_stat()
md = data_util.get_data(512, 32, *args, nw=4)
trn_iter = iter(md.trn_dl)

input('Press enter to start')
class_counter = [0] * 28
def count(i):
    class_counter[i] += 1
    return i

for i, bd in zip(range(100), trn_iter):
    # print(bd[0].shape, bd[1].shape)
    labels = bd[1].cpu().data.numpy()
    label_ids = [[count(i) for i, l in enumerate(row)
                  if l > 0.5]
                  for row in labels]
print(class_counter)
