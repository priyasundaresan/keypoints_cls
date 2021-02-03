import torch
import random
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pickle
import os
from datetime import datetime
import imgaug.augmenters as iaa

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

# Domain randomization
#transform = transforms.Compose([
#    iaa.Sequential([
#        iaa.AddToHueAndSaturation((-20, 20)),
#        iaa.LinearContrast((0.85, 1.2), per_channel=0.25), 
#        iaa.Add((-10, 30), per_channel=True),
#        iaa.GammaContrast((0.85, 1.2)),
#        iaa.GaussianBlur(sigma=(0.0, 0.6)),
#        iaa.ChangeColorTemperature((5000,35000)),
#        iaa.MultiplySaturation((0.95, 1.05)),
#        iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
#    ], random_order=True).augment_image,
#    transforms.ToTensor()
#])

class KeypointsDataset(Dataset):
    def __init__(self, dataset_folder, transform):
        self.transform = transform
        self.imgs = []
        self.labels = []
        img_folder = os.path.join(dataset_folder, 'images')
        labels_folder = os.path.join(dataset_folder, 'cls')
        for i in range(len(os.listdir(labels_folder))):
            try:
                label = np.load(os.path.join(labels_folder, '%05d.npy'%i))
                self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
                self.labels.append(torch.from_numpy(label).cuda())
            except:
                pass

    def __getitem__(self, index):  
        img = self.transform(cv2.imread(self.imgs[index]))
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    NUM_KEYPOINTS = 4
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    GAUSS_SIGMA = 10
    test_dataset = KeypointsDataset('/host/data/undo_reid_term/train/images',
                           '/host/data/undo_reid_term/train/actions', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
    img, gaussians = test_dataset[0]
    vis_gauss(gaussians)
 
