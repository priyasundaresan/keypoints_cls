import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/host/src')
from resnet_dilated import Resnet34_8s

class KeypointsGauss(nn.Module):
	def __init__(self, num_keypoints, img_height=480, img_width=640):
		super(KeypointsGauss, self).__init__()
		self.num_keypoints = num_keypoints
		self.img_height = img_height
		self.img_width = img_width
		self.fcn = Resnet34_8s(num_classes=num_keypoints)
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
		#start = time.time()
		x = self.fcn(x) 
		x = x.view(x.shape[0], self.num_keypoints, self.img_height*self.img_width)
		x = F.softmax(x, dim=1).double()
		#print(time.time() - start)
		return x

if __name__ == '__main__':
	model = KeypointsGauss(4).cuda()
	x = torch.rand((1,3,480,640)).cuda()
	result = model.forward(x)
	print(x.shape)
	print(result.shape)
	
