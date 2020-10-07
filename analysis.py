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

os.environ["CUDA_VISIBLE_DEVICES"]="2,7"

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
#keypoints.load_state_dict(torch.load('checkpoints/dr_time_GAUSS_KPTS_TIME/model_2_1_24_0.0021013115601769532.pth'))
keypoints.load_state_dict(torch.load('checkpoints/dr_time_GAUSS_KPTS_TIME/model_2_1_0_nan.pth'))

# cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

image_dir = 'data/dr_time/train/images'

episode_length = 16
heatmap_prev = torch.zeros([4, 480, 640]).cuda().double().unsqueeze(0)
for i in range(1, len(sorted(os.listdir(image_dir)))):
    pattern = '%05d.jpg'
    f = pattern%i
    f_prev = pattern%(i-1)
    if i%episode_length==0:
        heatmap_prev = torch.zeros([4, 480, 640]).cuda().double().unsqueeze(0)
        img_prev = np.zeros((480,640,3))
    else:
        img_prev = cv2.imread(os.path.join(image_dir, f_prev))
    img = cv2.imread(os.path.join(image_dir, f))
    img_tensor = transform(img).cuda().double().unsqueeze(0)
    img_prev_tensor = transform(img_prev).cuda().double().unsqueeze(0)
    inp = torch.cat((img_tensor, img_prev_tensor, heatmap_prev), 1).float()
    
    # GAUSS
    heatmap = prediction.predict(inp)
    heatmap_prev = heatmap.double()
    heatmap = heatmap.detach().cpu().numpy()
    prediction.plot(img, heatmap, image_id=i)
