import torch
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

transform = transforms.Compose([transforms.ToTensor()])

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False):
    U.unsqueeze_(1).unsqueeze_(2)
    V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    if normalize_dist:
        return normalize(G).double()
    return G.double()

def vis_gauss(gaussians):
    gaussians = gaussians.cpu().numpy()
    h1,h2,h3,h4 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)

class KeypointsDataset(Dataset):
    def __init__(self, dataset_folder, num_keypoints, img_height, img_width, transform, gauss_sigma=8):
        self.num_keypoints = num_keypoints
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = transform

        self.imgs = []
        self.classes = []
        self.keypoints = []
        labels_folder = os.path.join(dataset_folder, 'annots')
        img_folder = os.path.join(dataset_folder, 'images')
        for i in range(len(os.listdir(labels_folder))):
            label = np.load(os.path.join(labels_folder, '%05d.npy'%i), allow_pickle=True)
            kpt = label.item().get("kpt").reshape(num_keypoints, 2)
            cls = label.item().get("cls")
            self.classes.append(cls)
            kpt[:,0] = np.clip(kpt[:, 0], 0, self.img_width-1)
            kpt[:,1] = np.clip(kpt[:, 1], 0, self.img_height-1)
            self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
            self.keypoints.append(torch.from_numpy(kpt).cuda())

    def __getitem__(self, index):  
        img = self.transform(Image.open(self.imgs[index]))
        cls = self.classes[index]
        keypoints = self.keypoints[index]
        U = keypoints[:,0]
        V = keypoints[:,1]
        gaussians = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U, V)
        return img, gaussians, cls
    
    def __len__(self):
        return len(self.keypoints)

if __name__ == '__main__':
    NUM_KEYPOINTS = 4
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    GAUSS_SIGMA = 8 
    test_dataset = KeypointsDataset('/host/data/nonplanar_hulk_aug_kptcls/train', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
    img, gaussians, cls = test_dataset[0]
    print(img.shape, gaussians.shape, cls)
    vis_gauss(gaussians)
