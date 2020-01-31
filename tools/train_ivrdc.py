"""
Train Bi-directional Recurrent Neural Network.
Birnn and commensense iter 
commensense : thresh yingshe

"""
#CUDA_VISIBLE_DEVICES=2 nohup python -u ou_iter_v5.py --image_paths="prepare_data/vg/train/train_image_paths.json" --gt_file="prepare_data/vg/train/train_gt_file.pkl" --checkpoint_path="../Model/vg_ou_iter2_v5_0dot5/iter2" >../logs/vg_ou_iter2_v5_0dot5.txt 2>&1 &
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import gensim
import argparse
import cv2
import json
import math
import numpy as np
import cPickle as cp
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim
import os

# Trainning Parameters
train_batch_size = 50
dimension = 300
num_units = 128
train_learning_rate = 0.0001#0.000106#

predic_thresh = 0.05
train_keep_prob = 0.5
weight_decay = 0.0005#0.000005
#threshold = 0.1
def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()
	# image_paths file format: [ path of image_i for image_i in images ]
	# the order of images in image_paths should be the same with obj_dets_file
	parser.add_argument('--dataset',dest='dataset',help='dataset: vrd,vg,svg',default='vrd',type=str)
	# gt file format: [ gt_label, gt_box ]
	# 	gt_label: list [ gt_label(image_i) for image_i in images ]
	# 		gt_label(image_i): numpy.array of size: num_instance x 3
	# 			instance: [ label_s, label_r, label_o ]
	# 	gt_box: list [ gt_box(image_i) for image_i in images ]
	#		gt_box(image_i): numpy.array of size: num_instance x 2 x 4
	#			instance: [ [x1_s, y1_s, x2_s, y2_s], 
	#				    [x1_o, y1_o, x2_o, y2_o]]
	parser.add_argument('--iter_num',dest='iter_num', help='0: only birnn; 1: birnn_common; 2: iter2; ... ',default=1,type=int)
	parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='name of checkpoint path', default='', type=str)
	parser.add_argument('--threshold',dest='threshold',help='threshold for commonsense probability',default=0.1, type=float)
	parser.add_argument('--num_steps',dest='num_steps',help='number of step',default=10001, type=int)

	


	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args



def getUnionBBox(aBB, bBB, ih, iw):
	margin = 10
	return [max(0, min(aBB[0], bBB[0]) - margin), \
		max(0, min(aBB[1], bBB[1]) - margin), \
		min(iw, max(aBB[2], bBB[2]) + margin), \
		min(ih, max(aBB[3], bBB[3]) + margin)]

def getAppr(im, bb):
	subim = im[int(bb[1]) : int(bb[3]), int(bb[0]) : int(bb[2]), :]
	subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224/ subim.shape[0], interpolation=cv2.INTER_LINEAR)
	pixel_means = np.array([[[103.939, 116.779, 123.68]]])
	subim -= pixel_means
	return subim	

def getDualMask(ih, iw, bb):
	rh = 32.0 / ih
	rw = 32.0 / iw
	x1 = max(0, int(math.floor(bb[0] * rw)))
	x2 = min(32, int(math.ceil(bb[2] * rw)))
	y1 = max(0, int(math.floor(bb[1] * rh)))
	y2 = min(32, int(math.ceil(bb[3] * rh)))
	mask = np.zeros((32, 32))
	mask[y1 : y2, x1 : x2] = 1
	# print(x1, y1, x2, y2)
	# print(mask.sum(), (y2 - y1) * (x2 - x1))
	assert(mask.sum() == (y2 - y1) * (x2 - x1))
	return mask

def getDataset(tf_sub, tf_fea_vec, tf_obj, batch_size):
	tf_sub_reshape = tf.reshape(tf_sub, [batch_size, 1, 300])
	tf_fea_reshape = tf.reshape(tf_fea_vec, [batch_size, 1, 300])
	tf_obj_reshape = tf.reshape(tf_obj, [batch_size, 1, 300])
	datasets = tf.concat([tf_sub_reshape, tf_fea_reshape, tf_obj_reshape], 1)
	return datasets

def get_variables_to_train(trainable_scopes=None):
	"""Returns a list of variables to train.
	 Returns:
		A list of variables to train by the optimizer.
	"""
	if trainable_scopes is None:
		return tf.trainable_variables()
	else:
		scopes = trainable_scopes

	variables_to_train = []
	for scope in scopes:
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
		variables_to_train.extend(variables)
	return variables_to_train

#----------------------------Generate Train Data----------------------------#
args = parse_args()
print('Called with args:')
print(args)

threshold = args.threshold
if threshold==0.1:
	numb="0dot1"
elif threshold==0.3:
	numb="0dot3"
elif threshold==0.01:
	numb="0dot01"
elif threshold==0.5:
	numb='0dot5'
else:
	numb="random"
dataset_name = args.dataset
if dataset_name=="vrd":
	num_labels = 70
	max_num_det = 101
elif dataset_name=="vg":
	num_labels = 100
	max_num_det = 201
elif dataset_name=="svg":
	num_labels = 24
	max_num_det = 399
else:
	print("print correct dataset. vrd vg svg")

num_steps = args.num_steps
iter_num_ = args.iter_num

wv_model_dir = '../word2vec/300d_word2vec_'+dataset_name
obj_class_dir = 'prepare_data/'+dataset_name+'/object_class.txt'
pred_class_dir = 'prepare_data/'+dataset_name+'/predicate_class.txt'
gt_file_dir = 'prepare_data/'+dataset_name+'/train/train_gt_file.pkl'
image_paths_dir = 'prepare_data/'+dataset_name+'/train/train_image_paths.json'
proba_file_dir = '../outputs/all_kg_proba_for_'+dataset_name+'.pkl'
tensorboard_dir = '../tensorboard/'+dataset_name+'/iter'+str(iter_num_)
wv_model = gensim.models.Word2Vec.load(wv_model_dir)

f = open(obj_class_dir)
all_obj_class = f.readlines()
obj_class_num = len(all_obj_class)
f.close() 

fi = open(pred_class_dir, 'r')
pre_lines = fi.readlines()
pre_lines_num = len(pre_lines)
fi.close()


for i in xrange(obj_class_num):
	all_obj_class[i] = all_obj_class[i].strip().replace(' ', '-')

def load_train_data():
	print("loading gt file...")
	f = open(gt_file_dir, 'rb')
	dic = cp.load(f)
	all_train_gts = dic['gt_label']
	all_train_gt_bboxes = dic['gt_box']
	f.close()
    
	print("loading image paths...")
	f = open(image_paths_dir)
	train_img_paths = json.load(f)
	train_img_path = train_img_paths
	f.close()
	return all_train_gts, all_train_gt_bboxes, train_img_path

def load_proba():
	print("loading proba file...")
	f = open(proba_file_dir,'rb')
	all_kg_proba = cp.load(f)
	#print(all_kg_proba[38][88])
	if dataset_name=="vrd":
		all_kg_proba = np.array(all_kg_proba)
	f.close()
	return all_kg_proba

def Generate_input(all_train_gts, all_train_gt_bboxes, train_img_paths):
	num_train_img = len(train_img_paths)# 5
	print("prepare ",num_train_img," image data...")
	gt_bboxes = []
	gts = []
	img_paths = []
	for imgIdx in xrange(num_train_img):
		train_gts = all_train_gts[imgIdx]
		train_gt_bboxes = all_train_gt_bboxes[imgIdx]
		num_gts = len(train_gts)#train_gts.shape[0]
		for i in xrange(num_gts):
			gt_bboxes.append(train_gt_bboxes[i])
			gts.append(train_gts[i])
			img_paths.append(train_img_paths[imgIdx])

	return gts,gt_bboxes,img_paths


def Generate_batch_data(all_train_gts, all_train_gt_bboxes, train_img_paths,all_kg_proba,batch_size,begin_idx):
	train_pred = []
	train_bboxes = []
	train_ims = []
	train_poses = []
	train_subs = []
	train_objs = []
	train_labels = []
	train_proba = []

	gts = all_train_gts
	gt_bboxes = all_train_gt_bboxes
	im = cv2.imread(train_img_paths[begin_idx]).astype(np.float32, copy=False)
	ih = im.shape[0]
	iw = im.shape[1]
	for ii in xrange(batch_size):
		i = ii + begin_idx
		if train_img_paths[i] != train_img_paths[i-1] and i!=0:
			im = cv2.imread(train_img_paths[i]).astype(np.float32, copy=False)  
			ih = im.shape[0]
			iw = im.shape[1]
		else:
			im = im
			ih = ih
			iw = iw

		sub = gt_bboxes[i][0]
		obj = gt_bboxes[i][1]
		rBB = getUnionBBox(sub, obj, ih, iw)
		rAppr = getAppr(im, rBB)
		rMask = np.array([getDualMask(ih, iw, sub), getDualMask(ih, iw, obj)])
		train_ims.append(rAppr)
		train_poses.append(rMask)
		if dataset_name == "svg":
			sub_label = gts[i][0]+1
			obj_label = gts[i][2]+1
		else:
			sub_label = gts[i][0]
			obj_label = gts[i][2]
		sub_wv = wv_model[all_obj_class[sub_label - 1]]
		obj_wv = wv_model[all_obj_class[obj_label - 1]]
		pre_label = gts[i][1]
		pre_vec = np.zeros(num_labels)
		pre_vec[pre_label] = 1.0    # pre_label:[0, 69]
		if sub_label>=0 &obj_label>=0:
			#print(list(all_kg_proba[sub_label][obj_label].values()))
			if dataset_name=="vrd":
				pro = all_kg_proba[sub_label-1][obj_label-1]
			else:
				pro = list(all_kg_proba[sub_label-1][obj_label-1].values())
			tmp = np.array(pro)
			tmp[tmp<threshold] = 0.0
			pro = tmp.tolist()
		else:
			pro = np.zeros(num_labels)
			#print(pro)
		train_subs.append(sub_wv)
		train_objs.append(obj_wv)
		train_labels.append(pre_vec)
		train_proba.append(pro)
		if len(train_ims) == 0:
			continue

	train_ims = np.array(train_ims).astype(np.float32, copy=False)
	# train_ims = train_ims.transpose((0, 2, 3, 1))
	train_poses = np.array(train_poses).astype(np.float32, copy=False)
	train_poses = train_poses.transpose((0, 2, 3, 1))
	train_subs = np.array(train_subs).astype(np.float32, copy=False)
	train_objs = np.array(train_objs).astype(np.float32, copy=False)
	train_labels = np.array(train_labels)
	train_proba = np.array(train_proba)
    
	#print("data have done.")
	#print(train_proba.shape,train_proba.dtype,train_labels.shape,train_labels.dtype)
	return train_ims, train_poses, train_subs, train_objs, train_labels, train_proba


#-------------------------------------------Define Graph-----------------------------------------#

#---------------------------Feature Subnet---------------------------------#
def Feature_subnet(tf_ims, tf_poses, is_training, keep_prob, reuse):
	#--------------------Appearance Subnet---------------------#
	with tf.variable_scope('vgg_16', reuse=reuse):
		net = slim.repeat(tf_ims, 2, slim.conv2d, 64, [3, 3],
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='conv1')
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='conv2')
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='conv3')
		net = slim.max_pool2d(net, [2, 2], scope='pool3')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='conv4')
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='conv5')
		net = slim.max_pool2d(net, [2, 2], scope='pool5')

		# Use conv2d instead of fully_connected layers.
		net = slim.conv2d(net, 1024, [7, 7], 
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), padding='VALID', scope='fc_a1')
		net = slim.dropout(net, keep_prob, is_training=is_training,
		scope='dropout6')
		net = slim.conv2d(net, 1024, [1, 1], 
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='fc_a2')

		net = tf.reshape(net, [-1, 1024])
		# print(net.shape)
		fc_a = slim.fully_connected(net, 256,
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='fc_a3')	# [batch_size, 256]
		# fc_a = slim.fully_connected(net, 300,
		# 	weights_regularizer=slim.l2_regularizer(weight_decay),
		# 	biases_initializer=tf.constant_initializer(0), scope='fc_a3')	# [batch_size, 256]
		fc_a = tf.nn.dropout(fc_a, keep_prob)
        

	#----------------------Spatial Subnet----------------------#
	with tf.variable_scope('SpaNet', reuse=reuse): 
		# [batch, 16, 16, 96]
		conv1_p = slim.conv2d(tf_poses, 96, [5, 5], 2, 
			weights_regularizer=slim.l2_regularizer(weight_decay), scope='conv1_p')

		#[batch, 8, 8, 128]
		conv2_p = slim.conv2d(conv1_p, 128, [5, 5], 2, 
			weights_regularizer=slim.l2_regularizer(weight_decay), scope='conv2_p')

		# [batch, 1, 1, 64]
		conv3_p = slim.conv2d(conv2_p, 64, [8, 8], 1, 
			weights_regularizer=slim.l2_regularizer(weight_decay), padding='VALID', scope='conv3_p')

		conv3_p = tf.reshape(conv3_p, [-1, 1*1*64])

		fc = slim.fully_connected(conv3_p, 300,
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='fc')
		fc = tf.nn.dropout(fc, keep_prob)

	#----------------------Combine Subnets---------------------#
	with tf.variable_scope('CombNet', reuse=reuse):
		# Combine features
		concat = tf.concat([fc_a, conv3_p], 1)
		fc = slim.fully_connected(concat, 300,
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='fc')
		fc = tf.nn.dropout(fc, keep_prob)
		# fea_vec = tf.nn.softmax(fc1)
		fea_vec = tf.nn.relu(fc)

	return fea_vec

# def BiRNN(lstm_fw_cell, lstm_bw_cell, input_seq, weights, biases, batch_size):
# 	with tf.variable_scope('BRNN', reuse = reuse):
# 		init_state_fw = tf.zeros([batch_size, num_units])
# 		init_state_bw = tf.zeros([batch_size, num_units])
	
# 		(fw_outputs, bw_outputs), (fw_output_states, bw_output_states) = \
# 			tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, 
# 			input_seq, initial_state_fw=init_state_fw, 
# 			initial_state_bw=init_state_bw, dtype=tf.float32)

# 		result = tf.matmul(fw_output_states, weights['brnn_fw_out']) + tf.matmul(bw_output_states, weights['brnn_bw_out']) + biases['brnn_out']

# 	return result

def BiRNN(input_seq, reuse, batch_size):
	init_state_fw = tf.zeros([1, num_units])
	init_state_bw = tf.zeros([1, num_units])

	with tf.variable_scope('BRNN', reuse=reuse):
		w_initializer = tf.random_uniform_initializer(-1.0, 1.0)
		w_regularizer = slim.l2_regularizer(weight_decay)
		b_initializer = tf.zeros_initializer()

		# Weights
		with tf.variable_scope('weights', initializer=w_initializer, regularizer=w_regularizer):
			brnn_fw1_in_weight = tf.get_variable('brnn_fw1_in_weight', [num_units, dimension]) #128 300
			brnn_bw1_in_weight = tf.get_variable('brnn_bw1_in_weight', [num_units, dimension])
			brnn_fw2_in_weight = tf.get_variable('brnn_fw2_in_weight', [num_units, dimension])
			brnn_bw2_in_weight = tf.get_variable('brnn_bw2_in_weight', [num_units, dimension])
			brnn_fw3_in_weight = tf.get_variable('brnn_fw3_in_weight', [num_units, dimension])
			brnn_bw3_in_weight = tf.get_variable('brnn_bw3_in_weight', [num_units, dimension])

			brnn_fw_weight = tf.get_variable('brnn_fw_weight', [num_units, num_units]) #128 128
			brnn_bw_weight = tf.get_variable('brnn_bw_weight', [num_units, num_units])

			brnn_fw1_out_weight = tf.get_variable('brnn_fw1_out_weight', [num_units, num_labels])
			brnn_fw2_out_weight = tf.get_variable('brnn_fw2_out_weight', [num_units, num_labels])
			brnn_fw3_out_weight = tf.get_variable('brnn_fw3_out_weight', [num_units, num_labels])

			brnn_bw1_out_weight = tf.get_variable('brnn_bw1_out_weight', [num_units, num_labels])
			brnn_bw2_out_weight = tf.get_variable('brnn_bw2_out_weight', [num_units, num_labels])
			brnn_bw3_out_weight = tf.get_variable('brnn_bw3_out_weight', [num_units, num_labels])


		# Biases
		brnn_fw_bias = tf.get_variable('brnn_fw_bias', [num_units], initializer=b_initializer)
		brnn_bw_bias = tf.get_variable('brnn_bw_bias', [num_units], initializer=b_initializer)
		brnn_out_bias = tf.get_variable('brnn_out_bias', [num_labels], initializer=b_initializer)

		# print(tf.matmul(input_seq[:, 0, :], tf.transpose(brnn_fw1_in_weight, [1,0])).shape)
		state1_fw = tf.nn.relu(tf.matmul(input_seq[:, 0, :], tf.transpose(brnn_fw1_in_weight, [1,0]))
			+ tf.matmul(init_state_fw, brnn_fw_weight) + brnn_fw_bias)

		state2_fw = tf.nn.relu(tf.matmul(input_seq[:, 1, :], tf.transpose(brnn_fw2_in_weight, [1,0])) 
			+ tf.matmul(state1_fw, brnn_fw_weight) + brnn_fw_bias)

		state3_fw = tf.nn.relu(tf.matmul(input_seq[:, 2, :], tf.transpose(brnn_fw3_in_weight, [1,0])) 
			+ tf.matmul(state2_fw, brnn_fw_weight) + brnn_fw_bias)

		state1_bw = tf.nn.relu(tf.matmul(input_seq[:, 0, :], tf.transpose(brnn_bw1_in_weight, [1,0]))
			+ tf.matmul(init_state_bw, brnn_bw_weight) + brnn_bw_bias)

		state2_bw = tf.nn.relu(tf.matmul(input_seq[:, 1, :], tf.transpose(brnn_bw2_in_weight, [1,0]))
			+ tf.matmul(state1_bw, brnn_bw_weight) + brnn_bw_bias)

		state3_bw = tf.nn.relu(tf.matmul(input_seq[:, 2, :], tf.transpose(brnn_bw3_in_weight, [1,0]))
			+ tf.matmul(state2_bw, brnn_bw_weight) + brnn_bw_bias)

		# print(state1_fw.shape)
		# Outputs
		result = (tf.matmul(state1_fw, brnn_fw1_out_weight)
			+ tf.matmul(state2_fw, brnn_fw2_out_weight)
			+ tf.matmul(state3_fw, brnn_fw3_out_weight)) \
			+ (tf.matmul(state1_bw, brnn_bw1_out_weight) 
			+ tf.matmul(state2_bw, brnn_bw2_out_weight)
			+ tf.matmul(state3_bw, brnn_bw3_out_weight))\
			+ brnn_out_bias

	return result


def proba_conv(fP,keep_prob,reuse):
	fP = tf.reshape(fP, [batch_size,1,1,70])
	with tf.variable_scope('fP_conv', reuse=reuse): 
		# [batch, 16, 16, 96]
		conv1_fp = slim.conv2d(fF, 96, [5, 5], 2, 
			weights_regularizer=slim.l2_regularizer(weight_decay), scope='conv1_fp')
		print("conv1_fP: ",conv1_fp)
		#[batch, 8, 8, 128]
		conv2_fp = slim.conv2d(conv1_fp, 128, [5, 5], 2, 
			weights_regularizer=slim.l2_regularizer(weight_decay), scope='conv2_fp')
		print("conv2_fP: ",conv2_fp)
		# [batch, 1, 1, 64]
		conv3_fp = slim.conv2d(conv2_fp, 64, [8, 8], 1, 
			weights_regularizer=slim.l2_regularizer(weight_decay), padding='VALID', scope='conv3_fp')
		print("conv3_fP: ",vonv3_fp)
		#conv3_p = tf.reshape(conv3_p, [-1, 1*1*64])

		fc = slim.fully_connected(conv3_p, 70,
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_initializer=tf.constant_initializer(0), scope='fc_fp')
		fc = tf.nn.dropout(fc, keep_prob)

	return fP_conv

def _input(f_conv,reuse):
	xavier = tf.contrib.layers.variance_scaling_initializer()
	with tf.variable_scope("f_input",reuse=reuse):
		for i in xrange(2):
			c_input = slim.conv2d(f_conv,
									300,
									[1,1],
									weights_initializer=xavier,
									biases_initializer=tf.constant_initializer(0.0),
									scope="conv%02d"%i)
	return c_input

def comb_high_mid(fF,fea_vec,reuse,batch_size):
	initializer_fea_vec=tf.random_normal_initializer(mean=0.0, stddev=0.01)
   	initializer_fF=tf.random_normal_initializer(mean=0.0, stddev=0.01 * 1.)	
   	xavier = tf.contrib.layers.variance_scaling_initializer()
	fea_vec = tf.reshape(fea_vec, [batch_size, 1,1, 300])
	with tf.variable_scope('Combine_high_mid', reuse=reuse):
		map_prob = slim.fully_connected(fF, 
                                        300,
                                        weights_initializer=initializer_fF,
                                        biases_initializer=None,
                                        activation_fn=None,
                                        scope="map_prob")
		map_comp = tf.reshape(map_prob, [-1, 1, 1, 300], name="map_comp")
		fea_vec_comp = slim.conv2d(fea_vec, 
                                  300, 
                                  [1, 1], 
                                  weights_initializer=initializer_fea_vec,
                                  biases_initializer=tf.constant_initializer(0.0),
                                  activation_fn=None,
                                  scope="fea_vec_comp")
		fF_comb = tf.add(map_comp, fea_vec_comp, name="addition")
		fF_comb = tf.nn.relu(fF_comb, name="fF_comb")
 
		fF_conv = slim.conv2d(fF_comb, 300, [3, 3], 
                          activation_fn=tf.nn.relu, 
                          weights_initializer=xavier,
                          biases_initializer=tf.constant_initializer(0.0),
                          scope="conv_comb")
	fF_input = _input(fF_conv,reuse=reuse)
	return fF_input


def with_threshold(fP):
	fP_with_thr = []
	for i in xrange(len(fP)):
		if fP[i]<threshold:
			fP_with_thr.append(0.0)
		else:
			fP_with_thr.append(fP[i])
	return fP_with_thr

def deal_prob(fP):
	epsilon = 1e-5
	output = tf.clip_by_value(fP, epsilon, 1 - epsilon)
	output = tf.log(output)
	return output



def update_learningV_P(fF,fP,reuse,batch_size,name):
	update_iter = "P"
	mconv=3
	#initializer =  tf.uniform_unit_scaling_initializer(1.0)
	f_initializer =  tf.random_uniform_initializer(0.0,0.01)
	p_initializer =  tf.random_uniform_initializer(0.0,0.01)
	#print("fP: ",fP)
	with tf.variable_scope(update_iter,reuse=reuse):
		fP = tf.reshape(fP, [batch_size,1,1,num_labels])
		fF = tf.reshape(fF, [batch_size,1,1,num_labels])        
		f_update = slim.conv2d(fF,1,[mconv,mconv], weights_initializer=f_initializer, weights_regularizer=slim.l2_regularizer(weight_decay), scope='f_update'+name)
		p_update = slim.conv2d(fP,1,[mconv,mconv], weights_initializer=p_initializer, weights_regularizer=slim.l2_regularizer(weight_decay), scope='p_update'+name) 
		f_update_gate = tf.sigmoid(f_update+p_update,name="f_update_gate")
		equa1 = tf.multiply(f_update,fP, name="equa1")
		equa2 = tf.multiply(p_update,fF, name="equa2")
		result = tf.add(equa1,equa2,name="new_f")   
		result = tf.reshape(result,[batch_size,num_labels])
		equa1 = tf.reshape(equa1,[batch_size,num_labels])
		equa2 = tf.reshape(equa2,[batch_size,num_labels])
	return result,equa1,equa2,f_update_gate,f_update,p_update

def update_memory(fea_vec, fF_conv, reuse, batch_size, iter_num):
	memory_iter = 'Memory'#+str(iter_num)    
	with tf.variable_scope(memory_iter, reuse=reuse): 
		fF_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01)
		m_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.001*1.)
		fF_gate_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.001*1.)
		m_gate_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.001*1./1.)
        
		#initializer =  tf.random_uniform_initializer(0.0,0.01)
		#initializer =  tf.uniform_unit_scaling_initializer(1.0)
		mconv = 3
		fea_vec = tf.reshape(fea_vec, [batch_size, 1,1, 300])
		fF_conv = tf.reshape(fF_conv, [batch_size, 1,1, 300])
		#compute memory
		m_reset = slim.conv2d(fea_vec, 1, [mconv, mconv], weights_initializer=m_gate_initializer, weights_regularizer=slim.l2_regularizer(weight_decay), activation_fn=None,scope='reset_m')
		m_update = slim.conv2d(fea_vec, 1, [mconv, mconv], weights_initializer=m_gate_initializer, weights_regularizer=slim.l2_regularizer(weight_decay),activation_fn=None, scope='update_m')
        # compute fF
		f_input = slim.conv2d(fF_conv, 300, [mconv, mconv], weights_initializer=fF_initializer,biases_initializer=tf.constant_initializer(0.0), weights_regularizer=slim.l2_regularizer(weight_decay), activation_fn=None, scope='input_f')
		f_reset = slim.conv2d(fF_conv, 1, [mconv,mconv], weights_initializer=fF_gate_initializer,biases_initializer=tf.constant_initializer(0.0), weights_regularizer=slim.l2_regularizer(weight_decay), activation_fn=None, scope='reset_f')
		f_update = slim.conv2d(fF_conv, 1, [mconv,mconv], weights_initializer=fF_gate_initializer, biases_initializer=tf.constant_initializer(0.0), weights_regularizer=slim.l2_regularizer(weight_decay), activation_fn=None, scope='update_f')

      	# get the reset gate
		reset_gate = tf.sigmoid(f_reset + m_reset, name="reset_gate" )
		reset_res = tf.multiply(fea_vec, reset_gate, name="m_input_reset")
		m_input = slim.conv2d(reset_res, 300,[mconv,mconv], weights_initializer=m_initializer, biases_initializer=None, weights_regularizer=slim.l2_regularizer(weight_decay), activation_fn=None, scope='input_m')

      	Feat_new = tf.nn.relu(f_input+m_input,name="Feat_new")
      	update_gate = tf.sigmoid(f_update+m_update,name="update_gate")
      	fea_vec_diff = tf.multiply(update_gate, Feat_new-fea_vec, name="feature_diff" )
      	fea_vec_new = tf.add(fea_vec,fea_vec_diff,name="add" )
        #fea_vec_new = tf.add(update_gate*Feat_new,(1.0-update_gate)*fea_vec,name="add" )

	return fea_vec_new

def get_attend_confid(logits,name,reuse):
	#relu
	confid = slim.fully_connected(tf.nn.relu(logits), 1,\
			weights_regularizer=slim.l2_regularizer(weight_decay), \
				biases_initializer=tf.constant_initializer(0), scope=name,reuse=reuse)
	return confid

# Update weights for the target
def update_weights(labels, cls_prob):
	BETA = 0.5
	num_gt = labels.shape[0]
	index = np.arange(int(num_gt))
	cls_score = labels*cls_prob#cls_prob[index,labels]
	big_ones = cls_score >= 1. - BETA #True False
	#print("cls_score :: ",cls_score) # [?,70]
	weights = 1. - cls_score
	weights[big_ones] = BETA
	weights /= np.maximum(np.sum(weights), BETA)
	return weights


def aggregate_pred(confid,cls_score,name):
	with tf.variable_scope("Aggregate"+name):
		#confid <- cls_score
		print("AGGREGATE_PREED",confid,"\n",cls_score)        
		comb_confid = tf.stack(confid,axis=2,name="comb_confid")
		comb_attend = tf.nn.softmax(comb_confid,dim=2,name="comb_attend")
		comb_score = tf.stack(cls_score,axis=2, name='comb_score')
		
		att_cls_score = tf.reduce_sum(tf.multiply(comb_score, comb_attend, name='weighted_cls_score'), 
                                axis=2, name='attend_cls_score')
		att_cls_prob = tf.nn.softmax(att_cls_score,name="attend_cls_prob")
		att_cls_pred = tf.argmax(att_cls_score,axis=1, name="attend_cls_pred")
	return comb_score, comb_attend,att_cls_score,att_cls_prob
	
def add_memory_loss(weight,cls_score,attend_cls_score, label):
	cross_entropy = []
	iter_num=len(weight)
	print("cls_score: ",cls_score)
	print("attend_cls_score: ",attend_cls_score)
	with tf.variable_scope("cross_loss"):
		for it in xrange(iter_num):
			# loss
			cls_score_tmp = cls_score[it]
			ce_ins = tf.nn.softmax_cross_entropy_with_logits(logits=cls_score_tmp, labels=label,name="ce_ins_" + str(it))
			if it > 0:
				# cross entropy
				weight_tmp = weight[it-1]
				ce = tf.reduce_sum(tf.multiply(tf.transpose(weight_tmp), ce_ins, name="weight_"+ str(it)),name="ce_"+str(it))
			else:
				# loss 0
				ce = tf.reduce_mean(ce_ins, name="ce_"+str(it))
			cross_entropy.append(ce)
		
		ce_final = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=attend_cls_score,labels=label,name="ce_ins"),name="ce")
		loss = tf.add(ce_final,tf.reduce_mean(cross_entropy,name="cross_entropy"),name="loss")
	return loss


graph = tf.Graph()
with graph.as_default():
	# Input data
	tf_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	tf_test_batch_size = tf.placeholder(tf.int32, name='test_batch_size')
	
	# Train data
	tf_train_sub = tf.placeholder(tf.float32, shape=(train_batch_size, dimension), name='train_sub')
	tf_train_obj = tf.placeholder(tf.float32, shape=(train_batch_size, dimension), name='train_obj')
	tf_train_ims = tf.placeholder(tf.float32, shape=(train_batch_size, 224, 224, 3), name='train_ims')
	tf_train_poses = tf.placeholder(tf.float32, shape=(train_batch_size, 32, 32, 2), name='train_poses')
	tf_train_proba = tf.placeholder(tf.float32, shape=(train_batch_size, num_labels), name='train_probas')
	tf_train_labels = tf.placeholder(tf.float32, shape=(train_batch_size, num_labels), name='train_labels')

	# Test data
	tf_test_sub = tf.placeholder(tf.float32, shape=(None, dimension), name='test_sub')
	tf_test_obj = tf.placeholder(tf.float32, shape=(None, dimension), name='test_obj')
	tf_test_ims = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='test_ims')
	tf_test_poses = tf.placeholder(tf.float32, shape=(None, 32, 32, 2), name='test_poses')
	tf_test_proba = tf.placeholder(tf.float32, shape=(None, num_labels), name='test_probas')

	global_step = tf.Variable(0, trainable=False)

	# Training Computation
	
	tf_train_fea = Feature_subnet(tf_train_ims, tf_train_poses, True, train_keep_prob, False)
	attend_confid = []
	loss_weight = []
	train_cls_score = []
	train_cls_prob = []
	train_cls_predicate = []
	proba_map_output = []
	proba_maps = []
	#train_proba_tmp = proba_conv(tf_train_proba)
	#train_proba_tmp = deal_prob(tf_train_proba)
	train_proba_tmp = tf_train_proba
	feature_vec = tf_train_fea

	if iter_num_ == 0:
		iter_num = 1 #only feature module once
	else:
		iter_num = iter_num_
	for it in xrange(iter_num):
		#train_proba_tmp = tf_train_proba
		print(it)        
		# new prediction fF
		tf_train_dataset = getDataset(tf_train_sub, feature_vec, tf_train_obj, train_batch_size)
		fF_it = BiRNN(tf_train_dataset, tf.AUTO_REUSE, train_batch_size)
		fF_it_p = tf.nn.softmax(fF_it, name="cls_prob_it")
		confid_mem_it = get_attend_confid(fF_it,"confid_mem_it"+str(it),reuse=tf.AUTO_REUSE)
		
		loss_weight_it = tf.py_func(update_weights,[tf_train_labels, fF_it],tf.float32)
		# append
		attend_confid.append(confid_mem_it)
		loss_weight.append(loss_weight_it)
		train_cls_score.append(fF_it)
		train_cls_prob.append(fF_it_p)

		
		update_for_feature,equ1,equa2,_,_,_ = update_learningV_P(fF_it,train_proba_tmp,tf.AUTO_REUSE,train_batch_size,"_for_feature")
		update_for_common,_,_,_,_,_ = update_learningV_P(fF_it,train_proba_tmp,tf.AUTO_REUSE,train_batch_size,"_for_common")
		# update memory
		if it != iter_num-1:
			fF_comb = comb_high_mid(update_for_feature,feature_vec,tf.AUTO_REUSE,train_batch_size)
			new_train_fea = update_memory(feature_vec,fF_comb,tf.AUTO_REUSE,train_batch_size,it)
			feature_vec = new_train_fea

		train_proba_tmp = update_for_common
		confid_com_it = get_attend_confid(train_proba_tmp,"confid_com_it"+str(it),reuse=tf.AUTO_REUSE)
		loss_weight_com_it = tf.py_func(update_weights,[tf_train_labels, train_proba_tmp],tf.float32)
		train_proba_tmp_p = tf.nn.softmax(train_proba_tmp, name="cls_com_it")
		# append
		attend_confid.append(confid_com_it)
		loss_weight.append(loss_weight_com_it)
		train_cls_score.append(train_proba_tmp)
		train_cls_prob.append(train_proba_tmp_p)

	if iter_num_==0:
		train_score = fF_it
		train_prediction = fF_it_p
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=fF_it), name='loss')
		attend_prob = fF_it_p
	else:
		train_score = train_proba_tmp
		comb_score, comb_attend, attend_score, attend_prob = aggregate_pred(attend_confid,train_cls_score,"train")
		loss = add_memory_loss(loss_weight,train_cls_score,attend_score,tf_train_labels)
		train_prediction = train_proba_tmp_p

	tf.summary.scalar('loss', loss)
	tf.losses.add_loss(loss)

	reg_loss = tf.losses.get_regularization_loss()
	tf.summary.scalar('regularization loss', reg_loss)

	total_loss = tf.losses.get_total_loss()
	tf.summary.scalar('total loss', total_loss)

	merged = tf.summary.merge_all()

	# Optimizer

	print("ALL VARIABLES")
	for var in tf.trainable_variables():
		print(var)
        
	trainable_scope = [
		'vgg_16/fc_a1/weights',
		'vgg_16/fc_a1/biases',
		'vgg_16/fc_a2/weights',
		'vgg_16/fc_a2/biases',
		'vgg_16/fc_a3/weights',
		'vgg_16/fc_a3/biases',
		'SpaNet/conv1_p/weights',
		'SpaNet/conv1_p/biases',
		'SpaNet/conv2_p/weights',
		'SpaNet/conv2_p/biases',
		'SpaNet/conv3_p/weights',
		'SpaNet/conv3_p/biases',
		'SpaNet/fc/weights',
		'SpaNet/fc/biases',
		'CombNet/fc/weights',
		'CombNet/fc/biases',
		'BRNN/weights/brnn_fw1_in_weight',
		'BRNN/weights/brnn_bw1_in_weight',
		'BRNN/weights/brnn_fw2_in_weight',
		'BRNN/weights/brnn_bw2_in_weight',
		'BRNN/weights/brnn_fw3_in_weight',
		'BRNN/weights/brnn_bw3_in_weight',
		'BRNN/weights/brnn_fw_weight',
		'BRNN/weights/brnn_bw_weight',
		'BRNN/weights/brnn_fw1_out_weight',
		'BRNN/weights/brnn_fw2_out_weight',
		'BRNN/weights/brnn_fw3_out_weight',
		'BRNN/weights/brnn_bw1_out_weight',
		'BRNN/weights/brnn_bw2_out_weight',
		'BRNN/weights/brnn_bw3_out_weight',
		'BRNN/brnn_fw_bias',
		'BRNN/brnn_bw_bias',
		'BRNN/brnn_out_bias',
		'Memory/reset_m/weights',
		'Memory/reset_m/biases',
		'Memory/update_m/weights',
		'Memory/update_m/biases',
		'Memory/input_m/weights',
		'Memory/input_m/biases',
		'Memory/reset_f/weights',
		'Memory/reset_f/biases',
		'Memory/update_f/weights',
		'Memory/update_f/biases',
		'Memory/input_f/weights',
		'Memory/input_f/biases',
		'P/f_update/weights',
		'P/f_update/biases',
		'P/p_update/weights',
		'P/p_update/biases' 
	]
	var_list = get_variables_to_train(trainable_scope)
	
    
	learning_rate = tf.train.exponential_decay(train_learning_rate, global_step, 1000, 0.7, staircase='True', name='learning_rate') #0.7
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	#optimizer = tf.train.MomentumOptimizer(train_learning_rate,0.9)
	# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step, name='optimizer')
	grads = optimizer.compute_gradients(total_loss, var_list=var_list)
	apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    
	# Test Computation
	tf_test_fea = Feature_subnet(tf_test_ims, tf_test_poses, False, 1.0,True)
	test_cls_score = []
	test_cls_prob = []
	test_attend_confid = []

	#test_proba_tmp = deal_prob(tf_test_proba)
	test_proba_tmp = tf_test_proba
	test_feature_vec = tf_test_fea
	for it in xrange(iter_num):
		#test_proba_tmp = tf_test_proba        
		print(it)   
		# new prediction fF
		tf_test_dataset = getDataset(tf_test_sub, test_feature_vec, tf_test_obj, tf_test_batch_size)
		test_fF_it = BiRNN(tf_test_dataset, True, tf_test_batch_size)
		test_fF_it_p = tf.nn.softmax(test_fF_it, name="test_cls_prob_it"+str(it))
		test_confid_mem_it = get_attend_confid(test_fF_it,"confid_mem_it"+str(it),reuse=True)
		# append
		test_attend_confid.append(test_confid_mem_it)
		test_cls_score.append(test_fF_it)
		test_cls_prob.append(test_fF_it_p)      

		# new prediction fP
		#test_proba_map,_ = map_func_(test_proba_tmp,test_fF_it)
		test_update_for_feature,_,_,_,_,_ = update_learningV_P(test_fF_it,test_proba_tmp,True,tf_test_batch_size,"_for_feature")
		test_update_for_common,_,_,_,_,_ = update_learningV_P(test_fF_it,test_proba_tmp,True,tf_test_batch_size,"_for_common")

		# update memory
		if it != iter_num-1:
			test_fF_comb = comb_high_mid(test_update_for_feature,test_feature_vec,True,tf_test_batch_size)
			new_test_fea = update_memory(test_feature_vec,test_fF_comb,True,tf_test_batch_size,it)
			test_feature_vec = new_test_fea

		test_proba_tmp = test_update_for_feature#common
		test_confid_mem_com_it = get_attend_confid(test_fF_it,"confid_com_it"+str(it),reuse=True)
		test_proba_tmp_p = tf.nn.softmax(test_proba_tmp, name="test_cls_prob_it"+str(it))
		
		# append
		test_attend_confid.append(test_confid_mem_it)
		test_cls_score.append(test_proba_tmp)
		test_cls_prob.append(test_proba_tmp_p) 
		
	if iter_num_==0:
		test_score = test_fF_it
		test_prediction = tf.nn.softmax(test_fF_it, name='test_prediction')
	else:
		_, _,test_attend_score,test_attend_prob = aggregate_pred(test_attend_confid,test_cls_score,"test")
		test_prediction = tf.nn.softmax(test_attend_score, name='test_prediction')
	#test_combine_comm_score = update_learningV_P(test_attend_score,test_proba_tmp,0,reuse=True,1.0,tf_test_batch_size)
	#test_combine_comm_prob = tf.nn.softmax(combine_comm_score,name='train_prediction_comb')

	#test_combine_comm_prob = update_P(test_attend_prob,tf_test_proba,0.3,"test_combine_prob")
    
	# Saver
	train_saver = tf.train.Saver()
	saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=2)


#------------------------------------------Define Session---------------------------------------#
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

MAXIMUM = 10000000.0
topN = np.ones([5,2])*MAXIMUM  # [step,loss]
topN[:,0] = -1 # step = -1
# Start training
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    
	# Initialize the variables
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()
	
	# Load checkpoint
	exclusions = ['BRNN','SpaNet','CombNet','Memory','P']
	variables_to_restore = []
	for var in slim.get_model_variables():
		excluded = False
		for exclusion in exclusions:
			if var.op.name.startswith(exclusion):
				excluded = True
				break
		if not excluded:
			variables_to_restore.append(var)

	res = slim.assign_from_checkpoint_fn('../pretrain/vgg_16.ckpt', variables_to_restore, ignore_missing_vars=True)
	res(session)


	
	#train_saver.restore(session, '../Model/ou_iter2_v5/iter2-8000')
	#session.run(global_step.assign(0))
	

	print("Initialized.")
	train_writer = tf.summary.FileWriter(tensorboard_dir, session.graph)
	#------------------------------------------train-------------------------------------------#

	print("Loading data.")
	all_train_gts, all_train_gt_bboxes, train_img_paths = load_train_data()
	all_kg_proba = load_proba()
	input_train_gts,input_train_gt_bboxes, input_train_img_paths = Generate_input(all_train_gts, all_train_gt_bboxes, train_img_paths)
	all_instrance = len(input_train_gt_bboxes)
	batch_size = train_batch_size
	batch_num = all_instrance // batch_size+1
	print("batch_size: ",50, "  batch_num: ",batch_num, "  all_instrance: ",all_instrance)
	print("Begin Training.")
	for step in range(num_steps):
		offset = (step * batch_size) % (all_instrance - batch_size)
		begin_idx = offset
		batch_ims, batch_poses, batch_sub, batch_obj, batch_labels, batch_proba=Generate_batch_data(input_train_gts, input_train_gt_bboxes, input_train_img_paths,all_kg_proba,batch_size,begin_idx)
		feed_dict = {tf_train_sub : batch_sub, tf_train_ims : batch_ims, tf_train_poses : batch_poses,
				tf_train_obj : batch_obj, tf_train_proba: batch_proba, tf_train_labels : batch_labels,tf_keep_prob : 0.5}
		_, ts,l, each_prob,trainpinput,ap,gt,lr, predictions, summary = session.run([apply_gradient_op,train_score,loss,train_cls_prob, tf_train_proba,attend_prob,tf_train_labels, learning_rate, train_prediction, merged], feed_dict=feed_dict)

			#print(acc)
		if step % 100 == 0 or l < 1.0:
			print("Minibatch learning rate %f" % (lr))
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Minibatch accuracy(attention): %.1f%%" % accuracy(ap, batch_labels))
			#print("last score: ",ts)
			print("last result: ",predictions)
			#print("traininput: ",trainpinput)
			#print("gt label: ",gt)

			max_loss = np.max(topN[:,1])

			if l < max_loss:
				# logs directory
				max_index = np.where(topN[:,1] == max_loss)[0][0]
				max_step = topN[max_index,0]

				topN[max_index,0] = step
				topN[max_index,1] = l
				
				saver.save(session, args.checkpoint_path, global_step=step)
				print(args.checkpoint_path+'-'+str(step) + ' are saved')

				file_remove_list = []
#				os.join(args.checkpoint_path,)
				file_remove_list.append(args.checkpoint_path+'-'+str(int(max_step))+'.data-00000-of-00001')
				file_remove_list.append(args.checkpoint_path+'-'+str(int(max_step))+'.index')
				file_remove_list.append(args.checkpoint_path+'-'+str(int(max_step))+'.meta')
				for file_item in file_remove_list:
					if os.path.exists(file_item):
						print(file_item +' exists, so we remove it.')
						os.remove(file_item) 
						print(file_item + " deleted")
					elif max_step != -1:
						print(file_item + ' do not exist !')
			train_writer.add_summary(summary, step)

			
	saver.save(session, args.checkpoint_path, global_step=num_steps)
	print("Trained.")
	session.close()
