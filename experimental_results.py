import pickle
import colorsys
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

keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
#keypoints.load_state_dict(torch.load('checkpoints/real_aug_GAUSS_KPTS_ONLY/model_2_1_22_0.0024831097712612773.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/real_depth_aug_GAUSS_KPTS_ONLY/model_2_1_24_0.0026888263467603687.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/real_rgb_aug_GAUSS_KPTS_ONLY/model_2_1_24_0.0025266500706724905.pth'))
keypoints.load_state_dict(torch.load('checkpoints/real_rgbd_aug_GAUSS_KPTS_ONLY/model_2_1_24_0.0027264219537045022.pth'))

use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    torch.cuda.set_device(0)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

#image_dir = 'data/real_rgbd_aug/test/images'
#gt_keypoints_dir = 'data/real_rgb_aug/test/keypoints'
rgb_dir = 'data/real_test_set/images'
depth_dir = 'data/real_test_set/images_depth'
gt_keypoints_dir = 'data/real_test_set/keypoints'
output_dir = "preds_gt_pred"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

dense_tier3 = [0,2,3,6,7,8,9,10,12,13,14,28,29,42,49,50,78]
dense_tier2 = [1,4,5,11,22,33,34,35,36,37,38,43,44,45,47,48,51,52,53,54,55,56,57,58,59,60,61,62,65,66,67,68,69,70,71,72,73,75,76,77]
dense_tier1 = [23,24,25,26,27,40,41,46]

dense_tier1_error = np.array([0,0,0,0]).astype(float)
dense_tier2_error = np.array([0,0,0,0]).astype(float)
dense_tier3_error = np.array([0,0,0,0]).astype(float)

l_error = p_error = h_error = r_error = 0
for j, f in enumerate(sorted(os.listdir(rgb_dir))):
    img = cv2.imread(os.path.join(rgb_dir, '%05d.jpg'%j))
    depth_img = cv2.imread(os.path.join(depth_dir, '%05d.jpg'%j))
    gt_keypoints = np.load(os.path.join(gt_keypoints_dir, '%05d.npy'%j))
    img_t = transform(img)
    img_t = img_t.cuda()
    depth_img_t = transform(depth_img)
    depth_img_t = depth_img_t.cuda()
    inp = torch.cat((img_t, depth_img_t),0).float()
    print(inp.shape)
    #heatmap = prediction.predict(img_t)
    heatmap = prediction.predict(inp)
    heatmap = heatmap.detach().cpu().numpy()
    pred_keypoints = prediction.plot(img, heatmap, image_id=j)
    for i in range(len(gt_keypoints)):
        px, py = pred_keypoints[i].astype(int)
        gx, gy = gt_keypoints[i].astype(int)
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/gt_keypoints.shape[0], 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(img, (px, py), 4, (R,G,B), -1)
        R, G, B = int(100 * r), int(100 * g), int(100 * b)
        cv2.circle(img, (gx, gy), 4, (R,G,B), -1)
    cv2.imwrite(os.path.join(output_dir, 'pred_%05d.png'%j), img)
    l = np.linalg.norm(pred_keypoints[0] - gt_keypoints[0])
    p = np.linalg.norm(pred_keypoints[1] - gt_keypoints[1])
    h = np.linalg.norm(pred_keypoints[2] - gt_keypoints[2])
    r = np.linalg.norm(pred_keypoints[3] - gt_keypoints[3])
    combined = np.clip(np.array([l,p,h,r]), 0, 15)
    l,p,h,r = combined
    l_error += l
    p_error += p
    h_error += h
    r_error += r
    if j in dense_tier3:
        dense_tier3_error += combined
    if j in dense_tier2:
        dense_tier2_error += combined
    if j in dense_tier1:
        dense_tier1_error += combined
num_images = len(os.listdir(rgb_dir))
print("Overall Error for l,p,h,r:", l_error/num_images, p_error/num_images, h_error/num_images, r_error/num_images)
print("Tier3 Error for l,p,h,r:", dense_tier3_error/len(dense_tier3))
print("Tier2 Error for l,p,h,r:", dense_tier2_error/len(dense_tier2))
print("Tier1 Error for l,p,h,r:", dense_tier1_error/len(dense_tier1))
