import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
#from src.model_multi_headed import KeypointsGauss
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
#keypoints.load_state_dict(torch.load('checkpoints/dr_cable_cycles_6400_GAUSS_KPTS_ONLY/model_2_1_10_0.0029300788630979243.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/dr_cable_cycles_9K_GAUSS_KPTS_ONLY/model_2_1_18_0.003089638756832411.pth'))
keypoints.load_state_dict(torch.load('checkpoints/real_crop/model_2_1_24.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/cable_mask_dset_GAUSS_KPTS_ONLY/model_2_1_8_0.0033715498981842476.pth'))

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

#image_dir = 'data/overhead_hairtie_random_resized_larger'
image_dir = 'datasets/real_crop_test/images'
#image_dir = 'data/hairtie_overcrossing_resized_masks'
#image_dir = 'data/overhead_hairtie_resized_masks'
classes = {0: "Undo", 1:"Reidemeister", 2:"Terminate"}
for i, f in enumerate(sorted(os.listdir(image_dir))):
    img = np.load(os.path.join(image_dir, f), allow_pickle=True)
    print(img.shape)
    img_t = transform(img)
    img_t = img_t.cuda()
    # GAUSS
    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    prediction.plot(img, heatmap, image_id=i)
 
    #heatmap, cls = prediction.predict(img_t)
    #cls = torch.argmax(cls).item()
    #heatmap = heatmap.detach().cpu().numpy()
    #prediction.plot(img, heatmap, image_id=i, cls=cls, classes=classes)
