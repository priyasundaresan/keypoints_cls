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

def jitter_gaussians(gaussians, width, height, sigma, U, V, pixel_offset=5, alpha=0.1):
    U = U.reshape(len(U,))
    V = V.reshape(len(V,))
    U += torch.randint(-pixel_offset, pixel_offset, U.shape).cuda()
    V += torch.randint(-pixel_offset, pixel_offset, V.shape).cuda()
    noise = torch.rand(gaussians.shape).cuda().double()
    noise_gauss = gauss_2d_batch(width, height, sigma, U, V)
    noise = noise*noise_gauss
    jittered = gaussians + alpha*noise
    return jittered

def vis_gauss(gaussians):
    gaussians = gaussians.cpu().numpy()
    h1,h2,h3,h4 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)

class KeypointsDataset(Dataset):
    def __init__(self, img_folder, labels_folder, num_keypoints, img_height, img_width, transform, episode_length=16, gauss_sigma=8):
        self.num_keypoints = num_keypoints
        self.img_height = img_height
        self.img_width = img_width
        self.episode_length = episode_length
        self.gauss_sigma = gauss_sigma
        self.transform = transform

        self.imgs = []
        self.labels = []
        for i in range(len(os.listdir(labels_folder))):
            #label = np.load(os.path.join(labels_folder, '%05d.npy'%i))[:-2].reshape(num_keypoints, 2)
            label = np.load(os.path.join(labels_folder, '%05d.npy'%i)).reshape(num_keypoints, 2)
            label[:,0] = np.clip(label[:, 0], 0, self.img_width-1)
            label[:,1] = np.clip(label[:, 1], 0, self.img_height-1)
            self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
            self.labels.append(torch.from_numpy(label).cuda())
        

    def __getitem__(self, index):  
        img_t = self.transform(cv2.imread(self.imgs[index]))
        labels_t = self.labels[index]
        U_t = labels_t[:,0]
        V_t = labels_t[:,1]
        gaussians_t = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U_t, V_t)
        if index%self.episode_length == 0:
            img_prev = self.transform(np.zeros((self.img_height, self.img_width, 3)).astype(np.uint8))
            gaussians_prev = torch.zeros([4, 480, 640]).cuda().double()
            use_time_loss = 0
        else:
            img_prev = self.transform(cv2.imread(self.imgs[index-1]))
            labels_prev = self.labels[index-1]
            U_prev = labels_prev[:,0]
            V_prev = labels_prev[:,1]
            gaussians_prev = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U_prev, V_prev)
            #gaussians_prev = jitter_gaussians(gaussians_prev, self.img_width, self.img_height, self.gauss_sigma, U_prev, V_prev)
            use_time_loss = 1
        return img_t, gaussians_t, img_prev, gaussians_prev, use_time_loss
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    NUM_KEYPOINTS = 4
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    GAUSS_SIGMA = 10
    test_dataset = KeypointsDataset('/host/data/dr_cable_clean_flipped/train/images',
                           '/host/data/dr_cable_clean_flipped/train/keypoints', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
    img, gaussians, img_prev, gaussians_prev, _ = test_dataset[1]
    vis_gauss(gaussians_prev)
