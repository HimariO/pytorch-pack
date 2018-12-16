import os
import glob
from PIL import Image
import numpy as np
import multiprocessing as mp

def run(img_list):
    for img in img_list:
        img_name = os.path.basename(img)
        new_name = img_name.replace('.jpg', '.png')
        if os.path.exists(os.path.join('/home/himari/文件/deform-conv/scripts/protein_dataset/v18_data_png/', new_name)):
            continue
        ori_img = np.asarray(Image.open(img).resize([1024, 1024]))
        if len(ori_img.shape) == 3:
            if 'red' in img_name:
                ori_img = ori_img[..., 0]
            elif 'green' in img_name:
                ori_img = ori_img[..., 1]
            elif 'blue' in img_name:
                ori_img = ori_img[..., 2]
            else:
                np_img = ori_img.astype(np.float32)
                np_img[..., 0] *=  0.299
                np_img[..., 1] *=  0.587
                np_img[..., 2] *=  0.114
                np_img = np_img.sum(axis=-1).astype(np.uint8)
                ori_img = np_img[...]
        new_img = Image.fromarray(ori_img)
        new_img.save(os.path.join('/home/himari/文件/deform-conv/scripts/protein_dataset/v18_data_png/', new_name))

def main():
    worker = 16
    img_list = glob.glob('/home/himari/文件/deform-conv/scripts/protein_dataset/train/v18_data/*.jpg')
    split_list = [img_list[i::worker] for i in range(worker)]
    process = [mp.Process(target=run, args=[split_list[i]], daemon=True) for i in range(worker)]
    for p in process: p.start()
    for p in process: p.join()

main()
