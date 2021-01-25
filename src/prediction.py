import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import numpy as np

# @ PRIYA
class Prediction:
    def __init__(self, model, img_height, img_width, use_cuda):
        self.model = model
        self.img_height  = img_height
        self.img_width   = img_width
        self.use_cuda = use_cuda
        
    def predict(self, imgs):
        # img: torch.Tensor(3, height, width) 
        if len(imgs.shape) == 4:
            imgs = imgs.view(-1, imgs.shape[1], imgs.shape[2], imgs.shape[3])    
        elif len(imgs.shape) == 3:
            imgs = imgs.view(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
            
        heatmap = self.model.forward(Variable(imgs))
        return heatmap

    
    def plot(self, img, heatmap, image_id=0, cls=None, classes=None):
        print("Running inferences on image: %d"%image_id)
        all_overlays = []
        h = heatmap[0][0]
        vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.65, vis, 0.35, 0)
        cv2.imwrite('preds/out%04d.png'%image_id, overlay)
