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
from src.model import KeypointsGauss
from src.dataset import KeypointsDataset, transform
MSE = torch.nn.MSELoss()
bceLoss = nn.BCELoss

os.environ["CUDA_VISIBLE_DEVICES"]="4"

def forward(sample_batched, model):
    img_t, gt_gauss_t, img_prev, gauss_prev, use_time_loss = sample_batched
    img_t = Variable(img_t.cuda().double())
    img_prev = Variable(img_prev.cuda().double())
    inp = torch.cat((img_t, img_prev, gauss_prev), 1).float()
    pred_gauss = model.forward(inp).double()
    kpt_loss = nn.BCELoss()(pred_gauss, gt_gauss_t)
    idxs = torch.nonzero(use_time_loss)
    time_loss = nn.L1Loss()(pred_gauss[idxs]-gauss_prev[idxs], gt_gauss_t[idxs]-gauss_prev[idxs])
    alpha = beta = 0.5
    loss = alpha*kpt_loss + beta*time_loss
    return loss

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    for epoch in range(epochs):

        train_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(sample_batched, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.item()), end='')
            print('\r', end='')
        print('train loss:', train_loss / i_batch)
        
        test_loss = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            loss = forward(sample_batched, model)
            test_loss += loss.item()
        print('test loss:', test_loss / i_batch)
        if epoch%2 == 0:
            torch.save(keypoints.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '_' + str(test_loss/i_batch) + '.pth')

# dataset
workers=0
dataset_dir = 'dr_cable_cycles_6400'
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, dataset_dir+'_GAUSS_KPTS_TIME')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_dataset = KeypointsDataset('data/%s/train/images'%dataset_dir,
                           'data/%s/train/keypoints'%dataset_dir, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = KeypointsDataset('data/%s/test/images'%dataset_dir,
                           'data/%s/test/keypoints'%dataset_dir, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH).cuda()

# optimizer
#optimizer = optim.Adam(keypoints.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
optimizer = optim.Adam(keypoints.parameters(), lr=0.0001)

fit(train_data, test_data, keypoints, epochs=epochs, checkpoint_path=save_dir)
