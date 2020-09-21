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
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
#keypoints.load_state_dict(torch.load('checkpoints/dr_cable_clean_flipped_GAUSS_KPTS_ONLY/model_2_1_8_0.0029400087515022246.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/dr_cable_cycles_GAUSS_KPTS_ONLY/model_2_1_6_0.0029278998840831802.pth'))
keypoints.load_state_dict(torch.load('checkpoints/dr_cable_cycles_6400_GAUSS_KPTS_ONLY/model_2_1_6_0.0028304938931869726.pth'))

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

#image_dir = 'data/global_cable/images'
#image_dir = 'data/undo_reid_term_braid/test/images'
#image_dir = 'data/undo_reid_term_capsule/test/images'
image_dir = 'data/overhead_round_shoelace_resized'
#image_dir = 'data/overhead_hairtie_resized'
#image_dir = 'data/overhead_hairtie_random_fabric_resized'
#image_dir = 'data/overhead_hairtie_random_resized'
classes = {0: "Undo", 1:"Reidemeister", 2:"Terminate"}
for i, f in enumerate(sorted(os.listdir(image_dir))):
    img = cv2.imread(os.path.join(image_dir, f))
    print(img.shape)
    img_t = transform(img)
    img_t = img_t.cuda()
    # GAUSS
    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    prediction.plot(img, heatmap, image_id=i)
 
