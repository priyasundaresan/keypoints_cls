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

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
#keypoints.load_state_dict(torch.load('checkpoints/nonplanar_endpts_GAUSS_KPTS_ONLY/model_2_1_6_0.005121520109637978.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/nonplanar_endpts_GAUSS_KPTS_ONLY/model_2_1_10_0.005090024586942133.pth'))
keypoints.load_state_dict(torch.load('checkpoints/nonplanar_hulk_aug_multicolor_reannot/model_2_1_24_0.010199317085151347.pth'))

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

image_dir = 'data/nonplanar_hulk_aug/test/images'
classes = {0: "Undo", 1:"Reidemeister", 2:"Terminate"}
if not os.path.exists('preds'):
    os.mkdir('preds')
for i, f in enumerate(sorted(os.listdir(image_dir))):
    img = cv2.imread(os.path.join(image_dir, f))
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
