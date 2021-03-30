import numpy  as np
import cv2
from random import shuffle
import os

if __name__ == '__main__':
    dataset_dir = os.getcwd()
    inp_img_dir = os.path.join(dataset_dir, 'images_tworope')
    output_train_dir = os.path.join(dataset_dir, 'train')
    output_train_img_dir = os.path.join(output_train_dir, 'images')
    output_val_dir = os.path.join(dataset_dir, 'test')
    output_val_img_dir = os.path.join(output_val_dir, 'images')
    dirs = [output_train_dir, output_train_img_dir, output_val_dir, output_val_img_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    train_idx = 0
    val_idx = 0
    split_idx = int(len(os.listdir(inp_img_dir))*0.8)

    fnames = os.listdir(inp_img_dir)
    shuffle(fnames)
    for idx, f in enumerate(fnames):
        if 'jpg' in f:
            img = cv2.imread(os.path.join(inp_img_dir, f))
            if idx < split_idx:
                cv2.imwrite(os.path.join(output_train_img_dir, '%05d.jpg'%train_idx), img)
                train_idx += 1
            else:
                cv2.imwrite(os.path.join(output_val_img_dir, '%05d.jpg'%val_idx), img)
                val_idx += 1
