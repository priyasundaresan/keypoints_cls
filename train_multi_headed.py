import pickle
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
from src.model_multi_headed import KeypointsGauss
from src.dataset_multi_headed import KeypointsDataset, transform

MSE = torch.nn.MSELoss()
bceLoss = nn.BCELoss()
bceLogitsLoss= F.binary_cross_entropy_with_logits

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def forward(sample_batched, model):
    img, gt_gauss, cls = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    pred_gauss, cls_pred = model.forward(img)
    cls_loss = bceLogitsLoss(cls_pred.view(cls.shape), cls.float().cuda()).double()
    kpt_loss = (1-cls.item())*bceLoss(pred_gauss.double(), gt_gauss)
    return (1-kpt_loss_weight)*cls_loss, kpt_loss_weight*kpt_loss

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    for epoch in range(epochs):

        train_loss = 0.0
        train_kpt_loss = 0.0
        train_cls_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer_kpt.zero_grad()
            optimizer_cls.zero_grad()
            cls_loss, kpt_loss  = forward(sample_batched, model)
            kpt_loss.backward(retain_graph=True)
            cls_loss.backward(retain_graph=True)
            optimizer_kpt.step()
            optimizer_cls.step()
            train_loss += kpt_loss.item() + cls_loss.item()
            train_kpt_loss += kpt_loss.item()
            train_cls_loss += cls_loss.item()
            print('[%d, %5d] kpts loss: %.3f, cls loss: %.3f' % \
	           (epoch + 1, i_batch + 1, kpt_loss.item(), cls_loss.item()), end='')
            print('\r', end='')
        print('train kpt loss:', (1/kpt_loss_weight)*train_kpt_loss/i_batch)
        print('train cls loss:', np.sqrt((1/(1-kpt_loss_weight))*train_cls_loss/i_batch))
        
        #test_loss = 0.0
        #test_kpt_loss = 0.0
        #test_cls_loss = 0.0
        #for i_batch, sample_batched in enumerate(test_data):
        #    cls_loss, kpt_loss  = forward(sample_batched, model)
        #    test_loss += kpt_loss.item() + cls_loss.item()
        #    test_kpt_loss += kpt_loss.item()
        #    test_cls_loss += cls_loss.item()
        #torch.save(keypoints.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '.pth')

# dataset
workers=0
dataset_dir = 'undo_reid_term'
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, dataset_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_dataset = KeypointsDataset('/host/data/nonplanar_hulk_aug_kptcls/train', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

#test_dataset = KeypointsDataset('/host/data/nonplanar_hulk_aug_kptcls/test', NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
#test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_data = None

use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH).cuda()

# optimizer
optimizer_kpt = optim.Adam(keypoints.parameters(), lr=1e-4, weight_decay=1.0e-4)
optimizer_cls = optim.Adam(keypoints.parameters(), lr=1e-4, weight_decay=1.0e-4)

fit(train_data, test_data, keypoints, epochs=epochs, checkpoint_path=save_dir)
