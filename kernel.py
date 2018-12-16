from fastai.conv_learner import *
from fastai.dataset import *
from fastai.model import *
from fastai.imports import *
from fastai.transforms import *
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
from termcolor import colored
from torch.nn.modules.loss import BCEWithLogitsLoss

import ml_util
import torch_model
import data_util


def get_leaner(use_torch_model=None):
    if use_torch_model == 'inceptionresnetv2':
        sz = 256 #image size
        bs = 16  #batch size
        arch = inceptionresnet_2 #specify target architecture
        md = data_util.get_data(sz, bs, *dataset_stat(), nw=4)
        learner = torch_model.ConvLearnerV3.pretrained(arch, md, ps=0.5) #dropout 50%
        learner.models_path = 'inceptionresnetv2'
    elif use_torch_model:
        sz = 512 #image size
        bs = 8  #batch size
        arch = get_torch_model(use_torch_model)
        md = data_util.get_data(sz, bs, *dataset_stat(), nw=4)
        learner = torch_model.ConvLearnerV2.pretrained(arch, md, ps=0.5, clip=0.5) #dropout 50%
        learner.models_path = 'models_v2'
    else:
        sz = 512 #image size
        bs = 8  #batch size
        arch = resnet50 #specify target architecture
        md = data_util.get_data(sz, bs, *dataset_stat(), nw=8)
        learner = torch_model.ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
    return learner


def get_torch_model(model_name):
    model_name = model_name if model_name else 'se_resnext50_32x4d' # could be fbresnet152 or inceptionresnetv2
    model_cadene = pretrainedmodels.__dict__[model_name]
    return model_cadene


def dataset_stat(skip_analsic=True):
    # train_names = list({f[:36] for f in os.listdir(data_util.TRAIN)})
    label_csv = pd.read_csv(data_util.LABELS)

    train_names = [id for id in label_csv['Id']]
    test_names = list({f[:36] for f in os.listdir(data_util.TEST)})
    tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)

    if not skip_analsic:
        bs = 16
        sz = 256
        md = data_util.get_data(sz, bs, tr_n, val_n, test_names)

        x, y = next(iter(md.trn_dl))
        x.shape, y.shape

        # display_imgs(np.asarray(md.trn_ds.denorm(x)))
        x_tot = np.zeros(4)
        x2_tot = np.zeros(4)
        for x,y in iter(md.trn_dl):
            tmp =  md.trn_ds.denorm(x).reshape(16,-1)
            x = md.trn_ds.denorm(x).reshape(-1,4)
            x_tot += x.mean(axis=0)
            x2_tot += (x**2).mean(axis=0)

        channel_avr = x_tot/len(md.trn_dl)
        channel_std = np.sqrt(x2_tot/len(md.trn_dl) - channel_avr**2)
        print(channel_avr, channel_std)
    return tr_n, val_n, test_names


def plot_learning_rate(use_torch_model=None, load_weight=None):
    learner = get_leaner(use_torch_model=use_torch_model)
    learner.opt_fn = optim.Adam
    learner.clip = 1.0 #gradient clipping
    learner.crit = ml_util.FocalLoss()
    learner.metrics = [ml_util.acc]

    print(learner.summary)
    learner.lr_find()
    learner.sched.plot()


def train(use_torch_model=None, start_stage=None, load_weight=None):
    learner = get_leaner(use_torch_model=use_torch_model)
    learner.opt_fn = optim.Adam
    learner.clip = 1.0 #gradient clipping
    learner.crit = ml_util.FocalLoss()
    learner.metrics = [ml_util.acc]

    try:
        stages = ['checkpoint_1', 'checkpoint_2', 'checkpoint_3', 'checkpoint_4', 'checkpoint_5']
        if start_stage: stages = stages[stages.index(start_stage):]
        print(stages)
        print('_' * 100)
        lr = 1e-2
        lrs = np.array([lr / 8, lr / 3, lr])

        for si, stage in enumerate(stages):
            init_weight = si == 0 and load_weight
            if init_weight:
                learner.load(load_weight)
            if stage == 'checkpoint_1':
                learner.unfreeze()
                learner.fit(lr, 1, best_save_name='cp1_best')
                learner.save('checkpoint_1')
                print('-' * 100, 'checkpoint_1')
            else:
                learner.unfreeze()
                if stage == 'checkpoint_2':
                    if not init_weight:
                        try: learner.load('checkpoint_1')
                        except: pass
                    learner.fit(lrs / 4, 4, cycle_len=2, use_clr=(10, 20), best_save_name='cp2_best')
                    learner.save('checkpoint_2')
                    print('-' * 100, 'checkpoint_2')
                elif stage == 'checkpoint_3':
                    if not init_weight: learner.load('checkpoint_2')
                    learner.fit(lrs / 4, 2, cycle_len=4, use_clr=(10, 20), best_save_name='cp3_best')
                    learner.save('checkpoint_3')
                    print('-' * 100, 'checkpoint_3')
                elif stage == 'checkpoint_4':
                    if not init_weight: learner.load('checkpoint_3')
                    learner.fit(lrs / 16, 1, cycle_len=8, use_clr=(5, 20), best_save_name='cp4_best')
                    learner.save('checkpoint_4')
                elif stage == 'checkpoint_5':
                    if not init_weight: learner.load('checkpoint_4')
                    learner.fit(lrs / 16, 1, cycle_len=8, use_clr=(5, 20), best_save_name='cp5_best')
                    learner.save('checkpoint_5')
                else:
                    raise ValueError
        learner.save('ResNet34_256_2')
    except KeyboardInterrupt:
        learner.save('ResNet34_256_I')


def val(use_torch_model=None, load_weight='ResNet34_256_1', skip_val=False):
    learner = get_leaner(use_torch_model=use_torch_model)
    learner.load(load_weight)
    print(colored(f'Loading weight {load_weight}', color='green'))

    preds, y = learner.TTA(n_aug=16)
    preds = np.stack(preds, axis=-1)
    preds = ml_util.sigmoid_np(preds)
    pred = preds.max(axis=-1)

    th = ml_util.fit_val(pred,y)
    th[th < 0.1] = 0.1

    print('Thresholds: ',th)
    print('F1 macro: ',f1_score(y, pred>th, average='macro'))
    print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
    print('F1 micro: ',f1_score(y, pred>th, average='micro'))
    print('Fractions: ',(pred > th).mean(axis=0))
    print('Fractions (true): ',(y > th).mean(axis=0))
    print('_' * 100)
    print('Start generating submit predictions')

    preds_t, y_t = learner.TTA(n_aug=16, is_test=True)
    preds_t = np.stack(preds_t, axis=-1)
    preds_t = ml_util.sigmoid_np(preds_t)
    pred_t = preds_t.max(axis=-1) #max works better for F1 macro score

    th_t = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])
    print('Fractions: ',(pred_t > th_t).mean(axis=0))
    data_util.save_pred(learner, pred_t, th_t)
    lb_prob = [
         0.362397820,0.043841336,0.075268817,0.059322034,0.075268817,
         0.075268817,0.043841336,0.075268817,0.010000000,0.010000000,
         0.010000000,0.043841336,0.043841336,0.014198783,0.043841336,
         0.010000000,0.028806584,0.014198783,0.028806584,0.059322034,
         0.010000000,0.126126126,0.028806584,0.075268817,0.010000000,
         0.222493880,0.028806584,0.010000000
    ]

    th_t = ml_util.fit_test(pred_t,lb_prob)
    th_t[th_t < 0.1] = 0.1
    print('Thresholds: ',th_t)
    print('Fractions: ',(pred_t > th_t).mean(axis=0))
    print('Fractions (th = 0.5): ',(pred_t > 0.5).mean(axis=0))

    data_util.save_pred(learner, pred_t, th_t, 'protein_classification_f.csv')
    data_util.save_pred(learner, pred_t, th, 'protein_classification_v.csv')
    data_util.save_pred(learner, pred_t, 0.5,'protein_classification_05.csv')

    class_list = [8,9,10,15,20,24,27]
    for i in class_list:
        th_t[i] = th[i]
    data_util.save_pred(learner, pred_t, th_t, 'protein_classification_c.csv')

    labels = pd.read_csv(data_util.LABELS).set_index('Id')
    label_count = np.zeros(len(data_util.name_label_dict))
    for label in labels['Target']:
        l = [int(i) for i in label.split()]
        label_count += np.eye(len(data_util.name_label_dict))[l].sum(axis=0)
    label_fraction = label_count.astype(np.float) / len(labels)
    # label_count, label_fraction
    th_t = ml_util.fit_test(pred_t, label_fraction)
    th_t[th_t < 0.05] = 0.05
    print('Thresholds: ', th_t)
    print('Fractions: ', (pred_t > th_t).mean(axis=0))
    data_util.save_pred(learner, pred_t, th_t, 'protein_classification_t.csv')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    fire.Fire({
        'train': train,
        'val': val,
        'lr': plot_learning_rate,
    })
