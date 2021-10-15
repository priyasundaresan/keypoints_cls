import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import colorsys
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

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def expectation(self, d):
        width, height = d.T.shape
        d = d.T.ravel()
        d_norm = self.softmax(d)
        x_indices = np.array([i % width for i in range(width*height)])
        y_indices = np.array([i // width for i in range(width*height)])
        exp_val = [int(np.dot(d_norm, x_indices)), int(np.dot(d_norm, y_indices))]
        return exp_val
    
    def plot(self, img, heatmap, image_id=0, cls=None, classes=None):
        print("Running inferences on image: %d"%image_id)
        all_overlays = []
        for i in range(self.num_keypoints):
            h = heatmap[0][i]
            tmp = self.expectation(h)
            pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
            vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.65, vis, 0.35, 0)
            overlay = cv2.circle(overlay, (pred_x,pred_y), 4, (0,0,0), -1)
            all_overlays.append(overlay)
        result1 = cv2.vconcat(all_overlays[:self.num_keypoints//2])
        result2 = cv2.vconcat(all_overlays[self.num_keypoints//2:])
        result = cv2.hconcat((result1, result2))
        if cls is not None:
            label = classes[cls]
            cv2.putText(result, label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite('preds/out%04d.png'%image_id, result)

    def plot_combined(self, img, heatmap, image_id=0):
        print("Running inferences on image: %d"%image_id)
        all_overlays = []
        keypoints = []
        combined_heatmap = np.zeros_like(heatmap[0][0])
        for i in range(self.num_keypoints):
            h = heatmap[0][i]
            combined_heatmap += h
            pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
            keypoints.append((pred_x, pred_y))
        vis = cv2.normalize(combined_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.65, vis, 0.35, 0)
        for i, (u,v) in enumerate(keypoints):
            (r, g, b) = colorsys.hsv_to_rgb(float(i)/len(keypoints), 1.0, 1.0)
            R, G, B = int(255 * r), int(255 * g), int(255 * b)
            overlay = cv2.circle(overlay, (u,v), 3, (R,G,B), -1)
        cv2.imwrite('preds/out%04d.png'%image_id, overlay)
