import numpy as np
import torch,pdb
import torch.nn as nn
from ConvLSTM import *
from utils import *
import time
from torch.autograd import Variable

ConvLSTM_channel = 64
sequence_length = 10
Boxing_dir = '/home/zhang7/Boxing'
player_list = [[5,6],[7,8],[9,10],[11,12]]
batch_size = 10
total_folder = 4
sigma = 0.1
img_h = 60
img_w = 60
max_epoch = 20
lr_rate = 1e-4
weight_decay = 5e-4
eval_loss = 200
save_interval = 500

mymodel = model(ConvLSTM_channel, sequence_length)
mymodel.cuda()
train_pose_dataset = Pose_Dataset(Boxing_dir, sequence_length, player_list, total_folder,sigma,max_epoch,img_h,img_w)
test_pose_dataset = Pose_Dataset(Boxing_dir, sequence_length, [[13,14]], 1,sigma,1,img_h,img_w)
train_loader = torch.utils.data.DataLoader(dataset = train_pose_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
										   num_workers=4,pin_memory = True)
test_loader = torch.utils.data.DataLoader(dataset = test_pose_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False,
										   num_workers=4,pin_memory = True)
total_data_num = train_pose_dataset.total_data_num
data_iter = iter(train_loader)
loss_fn = nn.MSELoss()
loss_record = 0
time_stamp = time.time()
optimizer = torch.optim.Adam(mymodel.parameters(),lr = lr_rate, weight_decay = weight_decay)
for it in range(total_data_num):
	loss = 0
	data, label = data_iter.next()

	data = data.permute(1,0,4,2,3)
	data = Variable(data.cuda())
	label = label.permute(1,0,4,2,3)
	label = Variable(label.cuda())
	
	model_output = mymodel(data)
	loss = loss_fn(model_output, label)
	loss_record += loss.data[0]
	mymodel.zero_grad()
	loss.backward()
	optimizer.step()
	if it%eval_loss == 0:
		print('iteration: {}, loss = {}, time = {}'.format(it, loss_record/eval_loss, time.time()-time_stamp))
		loss_record = 0
		time_stamp = time.time()
	if it!=0 and (it % save_interval == 0 or it  == total_data_num - 1):
		test_loss = 0
		for data,label in test_loader:
			data = data.permute(1,0,4,2,3)
			data = Variable(data.cuda())
			label = label.permute(1,0,4,2,3)
			label = Variable(label.cuda())
			model_output = mymodel(data)
			loss = loss_fn(model_output, label)
			test_loss += loss.data[0]			
		torch.save(mymodel.state_dict(), 'model_it_{}'.format(it))
		print('test_loss:{},model saved'.format(test_loss)) 
