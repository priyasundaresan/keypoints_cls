import pickle
import numpy as np
import matplotlib.pyplot as plt
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
from src.dataset import KeypointsDataset, transform, normalize
MSE = torch.nn.MSELoss()
bceLoss = nn.BCELoss

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def expectation(d):
    b,n,w,h = d.shape
    d_flattened = d.view(b,n,w*h)
    x_indices = torch.Tensor([i%w for i in range(w*h)]).cuda().double()
    y_indices = torch.Tensor([i//w for i in range(w*h)]).cuda().double()
    mu_x = torch.zeros(b,n,1).cuda().double()
    mu_y = torch.zeros(b,n,1).cuda().double()
    for b_idx in range(b):
        for n_idx in range(n):
            mu_x[b_idx, n_idx] = torch.dot(d_flattened[b_idx, n_idx], x_indices)
            mu_y[b_idx, n_idx] = torch.dot(d_flattened[b_idx, n_idx], y_indices)
    return mu_x, mu_y

def forward(sample_batched, model, beta=3):
    img_t, gt_gauss_t, img_prev, gauss_prev, use_time_loss = sample_batched
    img_t = Variable(img_t.cuda().double())
    img_prev = Variable(img_prev.cuda().double())
    inp = torch.cat((img_t, img_prev, gauss_prev), 1).float()
    pred_gauss = model.forward(inp).double()
    b,n,w,h = pred_gauss.shape
    kpt_loss = nn.BCELoss()(pred_gauss, gt_gauss_t)

    # Normalize pred, gauss distributions
    pred_gauss_norm = pred_gauss.clone()
    gt_gauss_t_norm = gt_gauss_t.clone()
    for i in range(n):
        pred_gauss_norm[:,i] /= pred_gauss_norm[:,i].sum()
        gt_gauss_t_norm /= gt_gauss_t_norm[:,i].sum()
        gauss_prev /= gauss_prev[:,i].sum()

    # Compute the expectations
    pred_exp_x, pred_exp_y = expectation(pred_gauss_norm)
    gt_exp_x, gt_exp_y = expectation(gt_gauss_t_norm)
    gt_prev_exp_x, gt_prev_exp_y = expectation(gauss_prev)

    idxs = torch.nonzero(use_time_loss)
    if len(idxs):
        x_offset_loss = nn.L1Loss()(pred_exp_x[idxs] - gt_prev_exp_x[idxs], gt_exp_x[idxs] - gt_prev_exp_x[idxs])
        y_offset_loss = nn.L1Loss()(pred_exp_y[idxs] - gt_prev_exp_y[idxs], gt_exp_y[idxs] - gt_prev_exp_y[idxs])
        time_loss = 0.5*x_offset_loss + 0.5*y_offset_loss
        alpha = 1.0 - 10**(-beta)
    else:
        time_loss = torch.Tensor([0.0]).cuda().double()
        alpha = 1.0
    #try:
    #    print(alpha*kpt_loss.item(), (1-alpha)*time_loss.item())
    #except:
    #    print(alpha*kpt_loss, (1-alpha)*time_loss)

    loss = alpha*kpt_loss + (1-alpha)*time_loss
    return loss, kpt_loss, time_loss

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    train_losses, train_kpt_losses, train_time_losses = [], [], []
    test_losses, test_kpt_losses, test_time_losses = [], [], []
    #epoch_to_start_time_loss = int(0.5*epochs)

    for epoch in range(epochs):

        train_loss = train_kpt_loss = train_time_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss, kpt_loss, time_loss = forward(sample_batched, model, beta=(epochs-epoch))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_kpt_loss += kpt_loss.item()
            train_time_loss += time_loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.item()), end='')
            print('\r', end='')
        train_losses.append(train_loss / (i_batch+1))
        train_kpt_losses.append(train_kpt_loss / (i_batch+1))
        train_time_losses.append(train_time_loss / (i_batch+1))
        print('train loss:', train_loss / (i_batch+1))
        
        test_loss = test_kpt_loss = test_time_loss = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            loss, kpt_loss, time_loss = forward(sample_batched, model, beta=(epochs-epoch))
            test_loss += loss.item()
            test_kpt_loss += kpt_loss.item()
            test_time_loss += time_loss.item()
        print('test loss:', test_loss / (i_batch+1))
        test_losses.append(test_loss / (i_batch+1))
        test_kpt_losses.append(test_kpt_loss / (i_batch+1))
        test_time_losses.append(test_time_loss / (i_batch+1))
        if epoch%1 == 0:
            torch.save(keypoints.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '_' + str(test_loss/(i_batch+1)) + '.pth')
    return train_losses, test_losses, train_kpt_losses, test_kpt_losses, train_time_losses, test_time_losses

# dataset
workers=0
dataset_dir = 'dr_time'
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
optimizer = optim.Adam(keypoints.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
#optimizer = optim.Adam(keypoints.parameters(), lr=0.0001)

train_losses, test_losses, \
train_kpt_losses, test_kpt_losses, \
train_time_losses, test_time_losses  = fit(train_data, test_data, keypoints, epochs=epochs, checkpoint_path=save_dir)

plt.title("Total Training Loss over Time")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.plot(np.arange(len(train_losses)), train_losses)
plt.plot(np.arange(len(test_losses)), test_losses)
plt.legend(["train", "val"])
plt.savefig(os.path.join(save_dir, "total_loss.png"))
plt.clf()

plt.title("Keypoint Loss over Time")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.plot(np.arange(len(train_losses)), train_kpt_losses)
plt.plot(np.arange(len(test_losses)), test_kpt_losses)
plt.legend(["train", "val"])
plt.savefig(os.path.join(save_dir, "kpt_loss.png"))
plt.clf()

plt.title("Time Loss over Time")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.plot(np.arange(len(train_losses)), train_time_losses)
plt.plot(np.arange(len(test_losses)), test_time_losses)
plt.legend(["train", "val"])
plt.savefig(os.path.join(save_dir, "time_loss.png"))
plt.clf()
