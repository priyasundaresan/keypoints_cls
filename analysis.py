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

os.environ["CUDA_VISIBLE_DEVICES"]="4"

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
keypoints.load_state_dict(torch.load('checkpoints/nonplanar-hulk-conditioned-aug/model_2_1_24.pth'))
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
image_dir = 'data/nonplanar-hulk-conditioned-aug/test/images'
#image_dir = 'data/hairtie_overcrossing_resized_masks'
#image_dir = 'data/overhead_hairtie_resized_masks'

data_dir = "non_cond_label_test"
test_dataset = KeypointsDataset('data/%s/images'%data_dir,
                           'data/%s/annots'%data_dir, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

for i, f in enumerate(test_data):
    img_t = f[0]
    # GAUSS
    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    prediction.plot(img_t.detach().cpu().numpy(), heatmap, image_id=i)
