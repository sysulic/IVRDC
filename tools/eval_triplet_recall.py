#!/usr/bin/env python

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
	# det file format: [ det_label, det_box ]
	# 	det_label: list [ det_label(image_i) for image_i in images ]
	# 		det_label(image_i): numpy.array of size: num_instance x 6
	# 			instance: [ prob_s, prob_r, prob_o, label_s, label_r, label_o ]
	# 	det_box: list [ det_box(image_i) for image_i in images ]
	#		det_box(image_i): numpy.array of size: num_instance x 2 x 4
	#			instance: [ [x1_s, y1_s, x2_s, y2_s], 
	#				    [x1_o, y1_o, x2_o, y2_o]]
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

def computeArea(bb):
	return max(0, bb[2] - bb[0] + 1) * max(0, bb[3] - bb[1] + 1)

def computeIoU(bb1, bb2):
	ibb = [max(bb1[0], bb2[0]), \
		max(bb1[1], bb2[1]), \
		min(bb1[2], bb2[2]), \
		min(bb1[3], bb2[3])]
	iArea = computeArea(ibb)
	uArea = computeArea(bb1) + computeArea(bb2) - iArea
	return (iArea + 0.0) / uArea

def computeOverlap(detBBs, gtBBs):
	aIoU = computeIoU(detBBs[0, :], gtBBs[0, :])
	bIoU = computeIoU(detBBs[1, :], gtBBs[1, :])
	return min(aIoU, bIoU)		

def eval_recall(args):
	f = open(args.det_file, "r")
	dets, det_bboxes = cp.load(f)
	f.close()
	# f = open(args.gt_file, "r")
	# all_gts, all_gt_bboxes = cp.load(f)
	# f.close()
	f = open(args.gt_file,"rb")
	dic = cp.load(f)
	all_gts = dic['gt_label']
	# print all_gts
	all_gt_bboxes = dic['gt_box']
	# print all_gt_bboxes
	f.close()
    
	f = open("prepare_data/vrd/rlp_test_bool.pkl","rb")
	zeroshot_all = cp.load(f)
	flag = args.flag
    
	num_img = len(dets)
	#num_img = 3
	tp = []
	fp = []
	score = []
	total_num_gts = 0
	zeroshot_num_gts = 0
	for i in xrange(num_img):
		# gts = all_gts[i]
		gts = np.array(all_gts[i])
		# print gts
		# gt_bboxes = all_gt_bboxes[i]
		gt_bboxes = np.array(all_gt_bboxes[i])
		num_gts = gts.shape[0]
		total_num_gts += num_gts
		if flag==1:        
			zeroshot = zeroshot_all[i]
			zeroshot_num_gts += len(zeroshot)-sum(zeroshot) 
        
		gt_detected = np.zeros(num_gts)
		dets[i] = np.array(dets[i])       
		if isinstance(dets[i], np.ndarray) and dets[i].shape[0] > 0:
			det_score = np.log(dets[i][:, 0]) + np.log(dets[i][:, 1]) + np.log(dets[i][:, 2])
			# det_score = []
			# # print(len(dets[i]))   
			# for j in xrange(len(dets[i])): 
			# 	scores = dets[i][j, 0] * dets[i][j, 1] * dets[i][j, 2] 
			# 	det_score.append(scores)
			# det_score = np.array(det_score)
			# print(det_score.shape)
			inds = np.argsort(det_score)[::-1]
			if args.num_dets > 0 and args.num_dets < len(inds):
				inds = inds[:args.num_dets]
			top_dets = dets[i][inds, 3:]
			top_scores = det_score[inds]
			# top_det_bboxes = det_bboxes[i][inds, :]
			top_det_bboxes = np.array(det_bboxes[i][inds, :])
			num_dets = len(inds)
			for j in xrange(num_dets):
				ov_max = 0
				arg_max = -1
				for k in xrange(num_gts):
					if flag == 1 and zeroshot[k] == 1:
						continue                        
					if gt_detected[k] == 0 and top_dets[j, 0] == gts[k, 0] and top_dets[j, 1] == gts[k, 1] and top_dets[j, 2] == gts[k, 2]:                          
						ov = computeOverlap(top_det_bboxes[j, :, :], gt_bboxes[k, :, :])
						if ov >= args.ov_thresh and ov > ov_max:
							ov_max = ov
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
	print top_recall,zeroshot_num_gts

if __name__ == '__main__':
	args = parse_args()

	print('Called with args:')
	print(args)

	eval_recall(args)
	
