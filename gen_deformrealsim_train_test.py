import numpy  as np
import cv2
from random import shuffle
import os

if __name__ == '__main__':
    dataset_dir = os.getcwd()
    output_train_dir = os.path.join(dataset_dir, 'train')
    output_train_img_dir = os.path.join(output_train_dir, 'images')
    output_val_dir = os.path.join(dataset_dir, 'test')
    output_val_img_dir = os.path.join(output_val_dir, 'images')
    dirs = [output_train_dir, output_train_img_dir, output_val_dir, output_val_img_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    listOfFiles = list()
    inp_img_dirs = [os.path.join('data', 'sim_trajs', 'data_wiping_v4')]
    for d in inp_img_dirs:
        for (dirpath, dirnames, filenames) in os.walk(d):
            fnames = [os.path.join(dirpath, file) for file in filenames]
            fnames = [f for f in fnames if 'video' in f]
            listOfFiles += fnames

    n_total_images = 300

    train_idx = 0
    val_idx = 0
    split_idx = int(n_total_images*0.8)
    print(split_idx)

    shuffle(listOfFiles)
    for idx, f in enumerate(listOfFiles):
        img = cv2.imread(f)
        img = cv2.resize(img, (320,320))
        if idx < split_idx:
            cv2.imwrite(os.path.join(output_train_img_dir, '%05d.jpg'%train_idx), img)
            train_idx += 1
        else:
            cv2.imwrite(os.path.join(output_val_img_dir, '%05d.jpg'%val_idx), img)
            val_idx += 1
        if idx == n_total_images:
            break
