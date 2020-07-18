import torch
from torch.autograd import Variable
import torch.nn.functional as f
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import numpy as np

# @ PRIYA
class Prediction:
    def __init__(self, model, num_keypoints, img_height, img_width, use_cuda):
        self.model = model
        self.num_keypoints = num_keypoints
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

    def expectation(self, distribution):
        # TODO
        distribution = distribution.T
        width, height = distribution.shape
        flattened_dist = distribution.ravel()
        flattened_dist = flattened_dist/flattened_dist.sum()
        x_indices = np.array([i % width for i in range(width*height)])
        y_indices = np.array([i // width for i in range(width*height)])
        mu_hat = [int(np.dot(flattened_dist, x_indices)), int(np.dot(flattened_dist, y_indices))]
        return mu_hat
    
    def plot(self, img, heatmap, cls, cls_to_label, image_id=0):
        print("Running inferences on image: %d"%image_id)
        label = cls_to_label[cls]
        (h1,h2,h3,h4) = heatmap[0]
        h1 = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h2 = cv2.normalize(h2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h3 = cv2.normalize(h3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h4 = cv2.normalize(h4, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r1 = cv2.addWeighted(grayscale, 0.5, h1, 0.5, 0)
        r2 = cv2.addWeighted(grayscale, 0.5, h2, 0.5, 0)
        r3 = cv2.addWeighted(grayscale, 0.5, h3, 0.5, 0)
        r4 = cv2.addWeighted(grayscale, 0.5, h4, 0.5, 0)
        res1 = cv2.hconcat([r1,r4]) # endpoints (r1 = right, r4 = left)
        res2 = cv2.hconcat([r2,r3]) # pull, hold (r2 = pull, r3 = hold)
        result = cv2.vconcat([res2,res1])
        cv2.putText(result, label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite('preds/out%04d.png'%image_id, result)
