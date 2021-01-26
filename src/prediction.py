import torch
from torch.autograd import Variable
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
            
        heatmap, cls = self.model.forward(Variable(imgs))
        return heatmap, cls

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

    def plot_saliency(self, img, img_t, image_id):
        print(image_id)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #_, mask = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lower = np.array([0,0,40])
        upper = np.array([50,200,200])
        mask = cv2.inRange(rgb, lower, upper)


        img_t = img_t.view(-1, img_t.shape[0], img_t.shape[1], img_t.shape[2])
        img_t.requires_grad_()
        heatmap = self.model(img_t)
        heatmap_pull = heatmap[0][-3]
        heatmap_hold = (heatmap[0][-2]).detach().cpu().numpy()
        pred_y, pred_x = np.unravel_index(heatmap_hold.argmax(), heatmap_hold.shape)
        mask[:, np.arange(pred_x, 640)] = 0
        mask = cv2.circle(mask, (pred_x,pred_y), 15, 0, -1)
        ys, xs = np.where(mask<255)
        #cv2.imwrite('preds/mask%04d.png'%image_id, mask)

        score_max_idx = heatmap_pull.argmax()
        score_max = heatmap.flatten()[score_max_idx]
        score_max.backward(retain_graph=True)
        saliency, _ = torch.max(img_t.grad.data.abs(),dim=1)
        #saliency = torch.mean(img_t.grad.data.abs(),dim=1, keepdim=True)
        saliency = saliency.detach().cpu().numpy().squeeze()
        saliency[ys, xs] = 0
        saliency /= saliency.sum()
        vis = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        result = cv2.addWeighted(img, 0.35, vis, 0.65, 0)
        img_t.grad.data.zero_() 

        #all_overlays = []
        #for i in range(self.num_keypoints):
        #    heatmap_local = heatmap[0][i]
        #    score_max_idx = heatmap_local.argmax()
        #    score_max = heatmap.flatten()[score_max_idx]
        #    score_max.backward(retain_graph=True)
        #    #saliency, _ = torch.max(img_t.grad.data.abs(),dim=1)
        #    saliency = torch.mean(img_t.grad.data.abs(),dim=1, keepdim=True)
        #    saliency = saliency.detach().cpu().numpy().squeeze()
        #    saliency[ys, xs] = 0
        #    vis = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        #    #vis = cv2.bitwise_and(vis, vis, mask=mask)
        #    overlay = cv2.addWeighted(img, 0.35, vis, 0.65, 0)
        #    all_overlays.append(overlay)
        #    img_t.grad.data.zero_() 
        #result1 = cv2.vconcat(all_overlays[:self.num_keypoints//2])
        #result2 = cv2.vconcat(all_overlays[self.num_keypoints//2:])
        #result = cv2.hconcat((result1, result2))
        #cv2.putText(result, "Left Endpoint", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.putText(result, "Right Endpoint", (650, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.putText(result, "Hold", (650, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.putText(result, "Pull", (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #result = cv2.resize(result, (640,480))

        cv2.imwrite('preds/saliency%04d.png'%image_id, result)
    
    def plot(self, img, heatmap, image_id=0, cls=None, classes=None):
        print("Running inferences on image: %d"%image_id)
        all_overlays = []
        for i in range(self.num_keypoints):
            h = heatmap[0][i]
            #tmp = self.expectation(h)
            pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
            vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.65, vis, 0.35, 0)
            overlay = cv2.circle(overlay, (pred_x,pred_y), 4, (0,0,0), -1)
            all_overlays.append(overlay)
        result1 = cv2.vconcat(all_overlays[:self.num_keypoints//2])
        result2 = cv2.vconcat(all_overlays[self.num_keypoints//2:])
        result = cv2.hconcat((result1, result2))
        #cv2.putText(result, "Left Endpoint", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.putText(result, "Right Endpoint", (650, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.putText(result, "Hold", (650, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.putText(result, "Pull", (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if cls is not None:
            label = classes[cls]
            cv2.putText(result, label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        result =  cv2.resize(result, (640,240))
        cv2.imwrite('preds/out%04d.png'%image_id, result)

    def crop_pull_hold(self, img, heatmap, image_id=0, crop_width=60, crop_height=60):
        print("Running inferences on image: %d"%image_id)
        crops = []
        for i in [1,2]:
            h = heatmap[0][i]
            #tmp = self.expectation(h)
            pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
            crop = img[pred_y - crop_height//2:pred_y + crop_height//2, pred_x - crop_width//2:pred_x + crop_width//2]
            #crop = cv2.resize(crop, (100,100))
            crops.append(crop)
        cv2.imwrite('preds/out%04d_pull.png'%image_id, crops[0])
        cv2.imwrite('preds/out%04d_hold.png'%image_id, crops[1])
