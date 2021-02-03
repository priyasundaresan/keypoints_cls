import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import torchvision.models as models
sys.path.insert(0, '/host/src')
from resnet_dilated import Resnet34_8s

class KeypointsGauss(nn.Module):
	def __init__(self, num_keypoints, img_height=480, img_width=640):
		super(KeypointsGauss, self).__init__()
		self.num_keypoints = num_keypoints
		self.num_outputs = self.num_keypoints
		self.img_height = img_height
		self.img_width = img_width
		self.resnet = models.resnet18(pretrained=True)
		modules = list(self.resnet.children())[:-1]      # delete the last fc layer.
		self.resnet = nn.Sequential(*modules)
		self.resnet_out_dim = 512
		#self.dropout = nn.Dropout(0.5)
		self.linear = nn.Linear(self.resnet_out_dim, out_features=1)
	def forward(self, img):
		features = self.resnet(img)
		features = features.reshape(features.size(0), -1)
		features = self.linear(features)
		return features
if __name__ == '__main__':
	model = KeypointsGauss(4).cuda()
	x = torch.rand((1,3,480,640)).cuda()
	result = model.forward(x)
	print(x.shape)
	print(result.shape)
