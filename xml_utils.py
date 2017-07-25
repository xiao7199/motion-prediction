from xml.dom import minidom
import numpy as np
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def xml_parsing(file_name, p1_id, p2_id, total_joint = 18):

	doc = minidom.parse(file_name)
	Joint = doc.getElementsByTagName("Joint")
	ske = doc.getElementsByTagName("Skeleton")
	id_info = []
	
	for i in ske:
		id_info.append(int(i.getElementsByTagName('PlayerId')[0].firstChild.data))
	assert len(id_info) == 6
	assert p1_id in id_info
	assert p2_id in id_info
	Joint_info_p1 = {}
	Joint_info_p2 = {}
	counter = 0
	for j in Joint:
		joint_name = j.getElementsByTagName("JointType")[0].firstChild.data
		if 'Hand' in joint_name:
			continue
		Pos_3d = j.getElementsByTagName("Position")[0]
		x = float(Pos_3d.getElementsByTagName("X")[0].firstChild.data)
		y = float(Pos_3d.getElementsByTagName("Y")[0].firstChild.data)
		z = float(Pos_3d.getElementsByTagName("Z")[0].firstChild.data)
		state = j.getElementsByTagName("TrackingState")[0].firstChild.data
		if counter <  total_joint:
			Joint_info_p1[joint_name] = [x,y,z,state]
		else:
			Joint_info_p2[joint_name] = [x,y,z,state]
		counter += 1
	if id_info.index(p1_id) < id_info.index(p2_id):
		return Joint_info_p1,Joint_info_p2
	else:
		return Joint_info_p2,Joint_info_p1

def get_structure_info():
	name_codebook = [u'HipCenter', u'Head', u'HipLeft', u'KneeRight', u'ShoulderRight', \
				u'Spine', u'WristRight', u'AnkleLeft', u'KneeLeft', u'ElbowLeft', \
				u'ShoulderCenter', u'FootRight', u'WristLeft', u'HipRight', u'FootLeft', \
				u'ElbowRight',  u'AnkleRight', u'ShoulderLeft']
	joint_num = len(name_codebook)
	bones = np.zeros((joint_num,joint_num))
	bones[name_codebook.index('HipCenter'),name_codebook.index('Spine')] = 1
	bones[name_codebook.index('HipCenter'),name_codebook.index('HipLeft')] = 1
	bones[name_codebook.index('HipCenter'),name_codebook.index('HipRight')] = 1
	bones[name_codebook.index('Spine'),name_codebook.index('ShoulderCenter')] = 1
	bones[name_codebook.index('ShoulderCenter'),name_codebook.index('ShoulderRight')] = 1
	bones[name_codebook.index('ShoulderRight'),name_codebook.index('ElbowRight')] = 1
	bones[name_codebook.index('ElbowRight'),name_codebook.index('WristRight')] = 1
	bones[name_codebook.index('ShoulderCenter'),name_codebook.index('ShoulderLeft')] = 1
	bones[name_codebook.index('ShoulderLeft'),name_codebook.index('ElbowLeft')] = 1
	bones[name_codebook.index('ElbowLeft'),name_codebook.index('WristLeft')] = 1
	bones[name_codebook.index('ShoulderCenter'),name_codebook.index('Head')] = 1
	bones[name_codebook.index('HipRight'),name_codebook.index('KneeRight')] = 1
	bones[name_codebook.index('KneeRight'),name_codebook.index('AnkleRight')] = 1
	bones[name_codebook.index('AnkleRight'),name_codebook.index('FootRight')] = 1
	bones[name_codebook.index('HipLeft'),name_codebook.index('KneeLeft')] = 1
	bones[name_codebook.index('KneeLeft'),name_codebook.index('AnkleLeft')] = 1
	bones[name_codebook.index('AnkleLeft'),name_codebook.index('FootLeft')] = 1
	bones = bones + bones.T
	return name_codebook,bones

def plot_3d(pose_3d_dict):
	name_codebook, bones = get_structure_info()
	joint_num = len(pose_3d_dict.keys())
	x = np.zeros((joint_num,))
	y = np.zeros((joint_num,))
	z = np.zeros((joint_num,))
	
	for i in range(joint_num):
		pose_3d_coor = pose_3d_dict[name_codebook[i]]
		x[i] = pose_3d_coor[0]
		y[i] = pose_3d_coor[1]
		z[i] = pose_3d_coor[2]

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	for i in range(joint_num):
		ax.scatter(x[i], y[i], z[i])

	for i in range(joint_num):
		for j in range(joint_num):
			if bones[i,j] == 1:
				ax.plot([x[i],x[j]], [y[i],y[j]], [z[i],z[j]])
	max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
	ax.plot(x, z, 'r+', zdir='y', zs=1.5)
	mid_x = (x.max()+x.min()) * 0.5
	mid_y = (y.max()+y.min()) * 0.5
	mid_z = (z.max()+z.min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)
	plt.show()

def get_pose_numpy_array(pose_3d_dict, get_2d = False):
	joint_num = len(pose_3d_dict.keys())
	name_codebook, bones = get_structure_info()
	pose_3d = np.zeros((joint_num,3))
	counter = 0
	for key in name_codebook:
		pose_3d[counter,:] = pose_3d_dict[key][:3]
		counter += 1
	if get_2d:
		pose_3d = pose_3d[:,[0,2]]
	return pose_3d
	
if __name__ == '__main__':
	p1,p2 = xml_parsing('Skeleton 0.xml', 5,6)
	plot_3d(p1)