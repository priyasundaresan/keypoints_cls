import torch
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

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=True):
    U.unsqueeze_(1).unsqueeze_(2)
    V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    G=G.view(G.shape[0], G.shape[1]*G.shape[2])
    if normalize_dist:
        return normalize(G).double()
    return G.double()

class KeypointsDataset(Dataset):
    def __init__(self, img_folder, labels_folder, num_keypoints, img_height, img_width, transform, gauss_sigma=8):
        self.num_keypoints = num_keypoints
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = 8
        self.transform = transform

        self.imgs = []
        self.labels = []
        for i in range(len(os.listdir(labels_folder))):
            label = np.load(os.path.join(labels_folder, '%05d.npy'%i))[:-2].reshape(num_keypoints, 2)
            label[:,0] = np.clip(label[:, 0], 0, self.img_width-1)
            label[:,1] = np.clip(label[:, 1], 0, self.img_height-1)
            self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
            self.labels.append(torch.from_numpy(label).cuda())

    def __getitem__(self, index):  
        img = self.transform(Image.open(self.imgs[index]))
        labels = self.labels[index]
        U = labels[:,0]
        V = labels[:,1]
        gaussians = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U, V)
        return img, gaussians
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    NUM_KEYPOINTS = 4
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    test_dataset = KeypointsDataset('/host/data/undo_reid_term/train/images',
                           '/host/data/undo_reid_term/train/actions', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform=transform)
    img, gaussians = test_dataset[0]
    print(img.shape)
    print(gaussians.shape)
 
