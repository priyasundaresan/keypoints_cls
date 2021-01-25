import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from config import *
from src.model import KeypointsGauss
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction

from datetime import datetime
from PIL import Image
import numpy as np

# model
keypoints = KeypointsGauss(1, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, channels=3).cuda()
keypoints.load_state_dict(torch.load('checkpoints/two_rope_endpoints/model_2_1_24.pth'))

# cuda
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    torch.cuda.set_device(0)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

if not os.path.exists('preds'):
    os.mkdir('preds')

image_dir = '/host/data/two_rope_endpoints/train/images'
for i, f in enumerate(sorted(os.listdir(image_dir))):
    img = cv2.imread(os.path.join(image_dir, f))
    img_t = transform(img)
    img_t = img_t.cuda()
    # GAUSS
    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    prediction.plot(img, heatmap, image_id=i)
