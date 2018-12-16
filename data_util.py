from fastai.conv_learner import *
from fastai.dataset import *

import os
import glob

import pandas as pd
import numpy as np
import scipy.optimize as opt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data.sampler import WeightedRandomSampler
from sampler import ImbalancedDatasetSampler


PATH = './'
TRAIN = './input/train/'
TEST = './input/test/'
# LABELS = './input/train.csv'
LABELS = './input/train_merge_2.csv'
SAMPLE = './input/sample_submission.csv'

name_label_dict = {
    0:  'Nucleoplasm',
    1:  'Nuclear membrane',
    2:  'Nucleoli',
    3:  'Nucleoli fibrillar center',
    4:  'Nuclear speckles',
    5:  'Nuclear bodies',
    6:  'Endoplasmic reticulum',
    7:  'Golgi apparatus',
    8:  'Peroxisomes', # rare
    9:  'Endosomes', # rare
    10:  'Lysosomes', # rare
    11:  'Intermediate filaments',
    12:  'Actin filaments',
    13:  'Focal adhesion sites',
    14:  'Microtubules',
    15:  'Microtubule ends',  # rare
    16:  'Cytokinetic bridge',
    17:  'Mitotic spindle',
    18:  'Microtubule organizing center',
    19:  'Centrosome',
    20:  'Lipid droplets',
    21:  'Plasma membrane',
    22:  'Cell junctions',
    23:  'Mitochondria',
    24:  'Aggresome',
    25:  'Cytosol',
    26:  'Cytoplasmic bodies',
    27:  'Rods & rings'  # rare
}


def count_sample(csv):
    class_counter = [0] * 28
    classes_by_id = {}

    for img_id, cls in zip(csv['Id'], csv['Target']):
        classes_by_id[img_id] = [int(c) for c in cls.split(' ')]
    for k, v in classes_by_id.items():
        for cls_id in v:
            class_counter[cls_id] += 1

    # class_counter = [(i, e) for i, e in enumerate(class_counter)]
    return class_counter


def open_rgby(path, id): #a function that reads RGBY image
    id = id.replace('.png', '').replace('.jpg', '')
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    assert all([os.path.exists(os.path.join(path, id+'_'+color+'.png'))
                for color in colors]), f'Cant find image match id [{id}]'
    img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags).astype(np.float32) / 255
           for color in colors]
    return np.stack(img, axis=-1)


def get_data(sz, bs, tr_n, val_n, test_names, nw=16):
    #data augmentation
    print(' ' + '_' * 100)
    print(f'|+ Image Size: {sz}')
    print(f'|+ Batch Size: {bs}')
    print(f'|+ Train Samples: {len(tr_n)}')
    print(f'|+ Validate Samples: {len(val_n)}')
    print(f'|+ Test Samples: {len(test_names)}')
    print(f'|+ Number of workes: {nw}')
    print(' ' + '_' * 100)

    aug_tfms = [RandomRotate(180, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    cls_counts = count_sample(pd.read_csv(LABELS))

    #mean and std in of each channel in the train set
    stats = A([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])

    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                            aug_tfms=aug_tfms)
    ds = SampleImageData.get_ds(pdFilesDataset, (tr_n[: -(len(tr_n) % bs)], TRAIN),
                        (val_n, TRAIN), tfms, test=(test_names, TEST))
    samplers = [ImbalancedDatasetSampler(ds_single) if di < 4 else None for di, ds_single in enumerate(ds)]
    md = SampleImageData(PATH, ds, bs, num_workers=nw, classes=None, samplers=samplers)
    return md


def display_imgs(x):
    columns = 4
    bs = x.shape[0]
    rows = min((bs+3)//4,4)
    fig = plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        for j in range(columns):
            idx = i+j*columns
            fig.add_subplot(rows, columns, idx+1)
            plt.axis('off')
            plt.imshow((x[idx,:,:,:3]*255).astype(np.int))
    plt.show()


def save_pred(learner, pred, th=0.5, fname='protein_classification.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)

    sample_df = pd.read_csv(SAMPLE)
    sample_list = list(sample_df.Id)
    pred_dic = {key: value for key, value in zip(learner.data.test_ds.fnames, pred_list)}
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(fname, header=True, index=False)


class SampleImageData(ImageData):
    def __init__(self, path, datasets, bs, num_workers, classes, samplers=None):
        trn_ds,val_ds,fix_ds,aug_ds,test_ds,test_aug_ds = datasets
        self.path, self.bs, self.num_workers, self.classes = path, bs, num_workers, classes
        self.samplers = samplers

        dl_config = [
            (trn_ds, samplers is None),
            (val_ds,False),
            (fix_ds,False),
            (aug_ds,False),
            (test_ds,False),
            (test_aug_ds,False)
        ]
        if samplers:
            dl_config = [config + (sampler, ) for config, sampler in zip(dl_config, samplers)]
        else:
            dl_config = [config + (None, ) for config in dl_config]
        self.trn_dl, self.val_dl, self.fix_dl, self.aug_dl, self.test_dl,self.test_aug_dl = [
            self.get_dl(ds, shuf, sampler) for ds, shuf, sampler in dl_config
        ]

    def get_dl(self, ds, shuffle, sampler):
        if ds is None: return None
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=False, sampler=sampler)


class pdFilesDataset(FilesDataset):

    def __init__(self, fnames, image_path, transform):
        self.labels = pd.read_csv(LABELS).set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]

        get_img_id = lambda x: os.path.basename(x).replace('_green.jpg', '').replace('_red.jpg', '').replace('_yellow.jpg', '').replace('_blue.jpg', '').replace('_green.png', '').replace('_red.png', '').replace('_yellow.png', '').replace('_blue.png', '')
        self.img_to_dir = {get_img_id(p): os.path.dirname(p)
                            for p in glob.glob(os.path.join(image_path, '*.jpg'))}
        self.img_to_dir.update({get_img_id(p): os.path.dirname(p)
                            for p in glob.glob(os.path.join(image_path, '*.png'))})
        self.img_to_dir.update({get_img_id(p): os.path.dirname(p)
                            for p in glob.glob(os.path.join(image_path, '*/*.jpg'))})
        self.img_to_dir.update({get_img_id(p): os.path.dirname(p)
                            for p in glob.glob(os.path.join(image_path, '*/*.png'))})
        # import pdb; pdb.set_trace()
        super().__init__(fnames, transform, image_path)

    def get_x(self, i):
        img = self.open_rgby(self.path, self.fnames[i])
        return cv2.resize(img, (self.sz, self.sz),cv2.INTER_AREA)

    def get_y(self, i):
        if(self.path == TEST):
            return np.zeros(len(name_label_dict),dtype=np.int)
        else:
            labels = self.labels.loc[self.fnames[i]]['Target']
            return np.eye(len(name_label_dict), dtype=np.float)[labels].sum(axis=0)

    def get_raw_label(self, i):
        if(self.path == TEST):
            return []
        else:
            labels = self.labels.loc[self.fnames[i]]['Target']
            return labels

    def open_rgby(self, path, id): #a function that reads RGBY image
        id = id.replace('.png', '').replace('.jpg', '')

        if 'v18' not in self.img_to_dir[id]:
            color_layers = []
            for c in ['red', 'green', 'blue', 'yellow']:
                img_name = id + '_' + c + '.png'
                img_filename = os.path.join(self.img_to_dir[id], img_name)
                pil_img = Image.open(img_filename)
                np_img = np.array(pil_img)
                if len(np_img.shape) < 2: np_img = np.asarray(pil_img)

                if len(np_img.shape) == 2:
                    color_layers.append(np_img[:, :, np.newaxis])
                elif len(np_img.shape) == 3:
                    print(f'weird image {img_filename}')
                    color_layers.append(np_img.max(axis=-1))
                else:
                    raise ValueError(f'Unsupported shape image {img_filename}')
            return np.concatenate(color_layers, axis=-1) / 255
        elif 'v18' in self.img_to_dir[id]:
            color_layers = []
            for c in ['red', 'green', 'blue', 'yellow']:
                try:
                    img_name = id + '_' + c + '.jpg'
                    img_filename = os.path.join(self.img_to_dir[id], img_name)
                    np_img = np.array(Image.open(img_filename))

                    if len(np_img.shape) == 3:
                        if c == 'green':
                            np_img = np_img[..., 1]
                        elif c == 'blue':
                            np_img = np_img[..., 2]
                        elif c == 'red':
                            np_img = np_img[..., 0]
                        else:
                            np_img = np_img.astype(np.float32)
                            np_img[..., 0] *=  0.299
                            np_img[..., 1] *=  0.587
                            np_img[..., 2] *=  0.114
                            np_img = np_img.sum(axis=-1).astype(np.uint8)
                    color_layers.append(np_img[:, :, np.newaxis])
                except:
                    print(path, id, np_img.shape, img_filename)
            return np.concatenate(color_layers, axis=-1) / 255
        else:
            raise ValueError(f'Something went wrong! path={path}, id={id}')

    @property
    def is_multi(self):
        return True

    @property
    def is_reg(self):
        return True
    #this flag is set to remove the output sigmoid that allows log(sigmoid) optimization
    #of the numerical stability of the loss function

    def get_c(self): return len(name_label_dict) #number of classes
