#!/usr/bin/env python

# import _init_paths
# import _init_paths

import argparse

import time, os, sys
import json
import cv2
import cPickle as cp
import numpy as np
import math


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()
	# gt file format: [ gt_label, gt_box ]
	# 	gt_label: list [ gt_label(image_i) for image_i in images ]
	# 		gt_label(image_i): numpy.array of size: num_instance x 3
	# 			instance: [ label_s, label_r, label_o ]
	# 	gt_box: list [ gt_box(image_i) for image_i in images ]
	#		gt_box(image_i): numpy.array of size: num_instance x 2 x 4
	#			instance: [ [x1_s, y1_s, x2_s, y2_s], 
	#				    [x1_o, y1_o, x2_o, y2_o]]
	parser.add_argument('--gt_file', dest='gt_file',
						help='file containing gts',
						default=None, type=str)
	parser.add_argument('--num_dets', dest='num_dets',
						help='max number of detections per image',
						default=50, type=int)
	# det file format: [ det_label ]
	# 	det_label: list [ det_label(image_i) for image_i in images ]
	# 		det_label(image_i): numpy.array of size: num_instance x 6
	# 			instance: [ prob_r, label_s, label_r, label_o ]
	parser.add_argument('--det_file', dest='det_file', 
						help='file containing triplet detections',
						default=None, type=str)
	
	parser.add_argument('--min_overlap', dest='ov_thresh',
						help='minimum overlap for a correct detection',
						default=0.5, type=float)
	parser.add_argument('--flag', dest='flag',
						help='flag==1 means zero-shot',
						default=0, type=int)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args
	

def eval_recall(args):
	f = open(args.det_file, "r")
	dets = cp.load(f)
	
	f.close()

	f = open(args.gt_file,"rb")
	dic = cp.load(f)
	all_gts = dic['gt_label']
	f.close()

	f = open("prepare_data/vg/rlp_test_bool.pkl","rb")
	zeroshot_all = cp.load(f)
	flag = args.flag    

	num_img = len(dets)
	tp = []
	fp = []
	score = []
	total_num_gts = 0
	zeroshot_num_gts = 0    
	for i in xrange(num_img):
		gts = np.array(all_gts[i])
		num_gts = gts.shape[0]
		total_num_gts += num_gts
        
		zeroshot = zeroshot_all[i]
		zeroshot_num_gts += len(zeroshot)-sum(zeroshot) 

		gt_detected = np.zeros(num_gts)
		dets[i] = np.array(dets[i])
		#print(dets[i].shape[0])
		if isinstance(dets[i], np.ndarray) and dets[i].shape[0] > 0:
			det_score = dets[i][:,0]

			inds = np.argsort(det_score)[::-1]
			if args.num_dets > 0 and args.num_dets < len(inds):
				inds = inds[:args.num_dets]
			top_dets = dets[i][inds, 1:]
			top_scores = det_score[inds]
			# print(top_dets)
			#print(top_scores)
			num_dets = len(inds)
			for j in xrange(num_dets):
				arg_max = -1
				for k in xrange(num_gts):
					#print(top_dets[j,:])                    
					if gt_detected[k] == 0 and top_dets[j, 0] == gts[k, 0] and top_dets[j, 1] == gts[k, 1] and top_dets[j, 2] == gts[k, 2]:
						if flag ==1 and zeroshot[k]==1:                        
							continue                            
						arg_max = k
				if arg_max != -1:
					gt_detected[arg_max] = 1
					tp.append(1)
					fp.append(0)
				else:
					tp.append(0)
					fp.append(1)
				score.append(top_scores[j])
	score = np.array(score)
	tp = np.array(tp)
	fp = np.array(fp)
	inds = np.argsort(score)
	inds = inds[::-1]
	tp = tp[inds]
	fp = fp[inds]
	tp = np.cumsum(tp)
	fp = np.cumsum(fp)
	if flag==1:    
		recall = (tp + 0.0) / zeroshot_num_gts
	else:
		recall = (tp + 0.0) / total_num_gts
	top_recall = recall[-1] 
	print top_recall

if __name__ == '__main__':
	args = parse_args()

	print('Called with args:')
	print(args)

	eval_recall(args)
	
