import pickle
import cv2
import os
import torch
from torchvision import transforms
import time
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
#from src.model_multi_headed import KeypointsGauss
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="5"
# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
#keypoints.load_state_dict(torch.load('checkpoints/dr_cable_cycles_6400_GAUSS_KPTS_ONLY/model_2_1_10_0.0029300788630979243.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/dr_cable_cycles_9K_GAUSS_KPTS_ONLY/model_2_1_18_0.003089638756832411.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/dr_cable_cycles_9.5K_GAUSS_KPTS_ONLY/model_2_1_4_0.0031701664664068427.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/real_aug_GAUSS_KPTS_ONLY/model_2_1_22_0.0024831097712612773.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/real_aug_more_doubles_GAUSS_KPTS_ONLY/model_2_1_24_0.002664597477481971.pth'))

#keypoints.load_state_dict(torch.load('checkpoints/real_aug_dbl_reannot_GAUSS_KPTS_ONLY/model_2_1_22_0.0035997631694393776.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/real_aug_dbl_reannot_GAUSS_KPTS_ONLY/model_2_1_0_0.004250190804181906.pth'))

#keypoints.load_state_dict(torch.load('checkpoints/hulk_nonplanar/model_2_1_2_0.004509914506588016.pth'))
keypoints.load_state_dict(torch.load('checkpoints/hulk_nonplanar/model_2_1_24_0.0041214284476606255.pth'))

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

#image_dir = 'data/overhead_hairtie_random_resized_larger'
#image_dir = 'data/hairtie_overcrossing_resized'
#image_dir = 'data/overhead_hairtie_resized'
#image_dir = 'data/real_aug_more_doubles/test/images'
image_dir = 'data/nonplanar-blue-jpg'
#image_dir = 'data/double_knots'
#image_dir = 'data/overhead_hairtie_random_fabric_resized'
#image_dir = 'data/hairtie_overcrossing_resized_masks'
#image_dir = 'data/overhead_hairtie_resized_masks'
classes = {0: "Undo", 1:"Reidemeister", 2:"Terminate"}
for i, f in enumerate(sorted(os.listdir(image_dir))):
    img = cv2.imread(os.path.join(image_dir, f))
    img = cv2.resize(img, (640,480))
    img_t = transform(img)
    img_t = img_t.cuda()

    # GAUSS
    #prediction.plot_saliency(img, img_t, image_id=i)

    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    prediction.plot(img, heatmap, image_id=i)

    #prediction.crop_pull_hold(img, heatmap, image_id=i)
 
