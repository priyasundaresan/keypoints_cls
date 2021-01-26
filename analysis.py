import pickle
import cv2
import os
import torch
from torchvision import transforms
import time
from torch.utils.data import DataLoader
from config import *
#from src.model import KeypointsGauss
from src.model_multi_headed import KeypointsGauss
from src.dataset_multi_headed import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="5"
# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
keypoints.load_state_dict(torch.load('checkpoints/blue_cls_aug/model_2_1_24.pth'))

keypoints.eval()

# cuda
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    torch.cuda.set_device(0)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

workers=0
dataset_dir = 'blue_cls_aug'
test_dataset = KeypointsDataset('data/%s/%s'%(dataset_dir,'train'), NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)

test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

classes = {0: 'Unterminated', 1: 'Terminated'}
for i, f in enumerate(test_data):
    img = cv2.imread('data/%s/%s/images/%05d.jpg'%(dataset_dir, 'train', i))
    img_t = f[0].float().cuda()
    # GAUSS
    heatmap, cls = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    cls = torch.sigmoid(cls).detach().cpu().item()
    #cls = cls.detach().cpu().item()
    prediction.plot(img, heatmap, image_id=i, cls=int(cls), classes=classes)
 
