import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import pdb

class encoder_decoder_3d(nn.Module):
	def __init__(self, input_3d_shape, feature_base = 32,grid_point = 64):
		self.encoder = nn.Sequential(
			nn.Conv3d(1, feature_base, (8,3,3), stride = (4,1,1), padding = (4,1,1))
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(feature_base)
			nn.Conv3d(feature_base, feature_base*2, (8,3,3), stride = (4,1,1), padding = (4,1,1))
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(feature_base*2)
			nn.Conv3d(feature_base*2, feature_base*4, (8,3,3), stride = (4,1,1), padding = (4,1,1))
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(feature_base*4)
			nn.Conv3d(feature_base*4, feature_base*4, (4,3,3), stride = (2,1,1), padding = (1,1,1))
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(feature_base*4)
			)
		self.decoder = nn.Sequential(
			nn.ConvTranspose3d(feature_base*4, feature_base*4, (4,3,3), stride = (4,1,1), padding = (4,1,1))
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(feature_base*4)
			)

		
class Generator(nn.Module):
	def __init__(self, input_size_3d, input_size_2d, base_feature,joint_num, encoder_decoder_flag = False):
		self.encoder_decoder_flag = encoder_decoder_flag
		h,w,c =input_size_2d
		self.feature_2_fc = nn.Sequential(
			nn.Conv2d(c, base_feature, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(base_feature, base_feature*2, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(base_feature*, base_feature, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(base_feature, base_feature, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Linear(base_feature, base_feature*2)
			nn.ReLU(inplace=True),
			nn.Linear(base_feature*2, base_feature)
			)
		self.pose3d_2_fc = nn.Sequential(
			nn.Linear(input_size_3d, base_feature*2)
			nn.ReLU(inplace=True),
			nn.Linear(base_feature*2, base_feature*2)
			nn.ReLU(inplace=True),
			nn.Linear(base_feature*2, base_feature*2)
			nn.ReLU(inplace=True),
			nn.Linear(base_feature*, base_feature))
		self.concat_2_encoder = nn.Sequential(
			nn.Linear(base_feature*2, base_feature*2)
			nn.ReLU(inplace=True),
			nn.Linear(base_feature*2, base_feature*2)
			nn.ReLU(inplace=True),
			nn.Linear(base_feature*2, base_feature*2)		
			)

	def forward(self, input_2d, input_3d):
		feature_2d = self.feature_2_fc(input_2d)
		feature_3d = self.feature_3_fc(input_3d)

