"""
Test Bi-directional Recurrent Neural Network.
"""

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

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    # image_paths file format: [ path of image_i for image_i in images ]
    # the order of images in image_paths should be the same with obj_dets_file
    parser.add_argument('--dataset',dest='dataset',help='dataset: vrd,vg,svg',default='vrd',type=str)
    parser.add_argument('--model',dest='model',help='model: XXX/XX',default=None,type=str)
    parser.add_argument('--output_name',dest='output_name',help='test_predica_...',default='test_predicate_default',type=str)
    parser.add_argument('--predic_type',dest='predic_type',help='1 test_prediction 2 attend_prob  3 attend_score',default=2, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



args = parse_args()
print('Called with args:')
print(args)

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

wv_model_dir = '../word2vec/300d_word2vec_'+dataset_name
obj_class_dir = 'prepare_data/'+dataset_name+'/object_class.txt'
pred_class_dir = 'prepare_data/'+dataset_name+'/predicate_class.txt'
gt_file_dir = 'prepare_data/'+dataset_name+'/test/test_gt_file.pkl'
image_paths_dir = 'prepare_data/'+dataset_name+'/test/test_image_paths.json'
proba_file_dir = '../outputs/all_kg_proba_for_'+dataset_name+'.pkl'
obj_dets_dir = 'prepare_data/'+dataset_name+'/obj_dets_file_test_for_'+dataset_name+'.pkl'
#obj_dets_file_for_vg_new
wv_model = gensim.models.Word2Vec.load(wv_model_dir)
predic_type = args.predic_type

output_name = args.output_name
model_name = args.model
# Trainning Parameters


# Trainning Parameters
test_batch_size = 50
dimension = 300
num_units = 128
predic_thresh = 0.05

def getPred(pred, max_num_det):
    if pred.shape[0] == 0:
        return pred
    inds = np.argsort(pred[:, 4])
    inds = inds[::-1]
    if len(inds) > max_num_det:
        inds = inds[:max_num_det]
    return pred[inds, :]

def getUnionBBox(aBB, bBB, ih, iw):
    margin = 10
    return [max(0, min(aBB[0], bBB[0]) - margin), \
        max(0, min(aBB[1], bBB[1]) - margin), \
        min(iw, max(aBB[2], bBB[2]) + margin), \
        min(ih, max(aBB[3], bBB[3]) + margin)]

def getAppr(im, bb):
    subim = im[int(bb[1]) : int(bb[3]), int(bb[0]) : int(bb[2]), :]
    subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0], interpolation=cv2.INTER_LINEAR)
    pixel_means = np.array([[[103.939, 116.779, 123.68]]])
    subim -= pixel_means
    # subim = subim.transpose((2, 0, 1))
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

# Test Data
wv_model = gensim.models.Word2Vec.load(wv_model_dir)

f = open(obj_class_dir)
all_obj_class = f.readlines()
obj_class_num = len(all_obj_class)
f.close() 
    
for i in xrange(obj_class_num):
    all_obj_class[i] = all_obj_class[i].strip().replace(' ', '-')


def load_test_data():
    f = open(obj_dets_dir)
    # [[[x1,y1,x2,y2,pro,label],...,[x1,y1,x2,y2,pro,label]]
    #  [[x1,y1,x2,y2,pro,label],...,[x1,y1,x2,y2,pro,label]]
    #  ...
    #  ]
    all_test_dets = cp.load(f)
    f.close()
    all_test_dets = np.array(all_test_dets)

    f = open(image_paths_dir)
    test_img_paths = json.load(f)
    f.close()

    return all_test_dets, test_img_paths

def load_proba():
    f = open(proba_file_dir,'rb')
    all_kg_proba = cp.load(f)
    f.close()
    #all_kg_proba = np.array(all_kg_proba)
    return all_kg_proba

# Start testing
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    
    
    saver = tf.train.import_meta_graph('../Model/'+model_name+'.meta')
    # saver.restore(session, tf.train.latest_checkpoint('../model'))
    session.run(tf.global_variables_initializer())
    
    saver.restore(session, '../Model/'+model_name)
    
    print("Begin Testing.")
    # Get input tensor
    tf_test_sub = tf.get_default_graph().get_tensor_by_name('test_sub:0')
    tf_test_obj = tf.get_default_graph().get_tensor_by_name('test_obj:0')
    tf_test_ims = tf.get_default_graph().get_tensor_by_name('test_ims:0')
    tf_test_poses = tf.get_default_graph().get_tensor_by_name('test_poses:0')
    tf_keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
    tf_test_batch_size = tf.get_default_graph().get_tensor_by_name('test_batch_size:0')
    tf_test_proba = tf.get_default_graph().get_tensor_by_name('test_probas:0')

    if predic_type==1:
        test_prediction = session.graph.get_tensor_by_name('test_prediction:0')
    elif predic_type==2:
        test_prediction = session.graph.get_tensor_by_name('Aggregatetest/attend_cls_prob:0')
    elif predic_type==3:
        test_prediction = session.graph.get_tensor_by_name('Aggregatetest/attend_cls_score:0')
    elif predic_type==4:
        test_prediction = session.graph.get_tensor_by_name('test_prediction_c:0')

    all_test_dets, test_img_paths = load_test_data()
    all_kg_proba = load_proba()
    pred = []
    pred_bboxes = []
    num_test_img = len(test_img_paths)
    for imgIdx in xrange(num_test_img):
        #print(test_img_paths[imgIdx])
        im = cv2.imread(test_img_paths[imgIdx]).astype(np.float32, copy=False)
        ih = im.shape[0]
        iw = im.shape[1]
        if imgIdx % 100 == 0:
            print("Image Index: ", imgIdx)
        all_test_dets[imgIdx] = np.array(all_test_dets[imgIdx])
        dets = getPred(all_test_dets[imgIdx], max_num_det)
        num_dets = dets.shape[0]
        pred.append([])
        pred_bboxes.append([])
        # print(dets.shape, num_dets)
        for subIdx in xrange(num_dets):
            ims = []
            poses = []
            subs = []
            objs = []
            pros = []
            for objIdx in xrange(num_dets):
                if subIdx != objIdx:
                    sub_label = int(dets[subIdx, 5])
                    obj_label = int(dets[objIdx, 5])
                    # print(sub_label, type(sub_label))
                    # print(obj_label, type(obj_label))
                    sub_wv = wv_model[all_obj_class[sub_label - 1]]
                    obj_wv = wv_model[all_obj_class[obj_label - 1]]
                    subs.append(sub_wv)
                    objs.append(obj_wv)
                    
                    sub = dets[subIdx, 0:4]
                    obj = dets[objIdx, 0:4]
                    rBB = getUnionBBox(sub, obj, ih, iw)
                    # print(rBB[0], rBB[1], rBB[2], rBB[3])
                    if sub_label>=0 &obj_label>=0:
                        #pro = all_kg_proba[sub_label-1][obj_label-1]
                        if dataset_name=="vrd":
                            pro = all_kg_proba[sub_label-1][obj_label-1]
                        else:
                            pro = list(all_kg_proba[sub_label-1][obj_label-1].values())
                    else:
                        pro = np.zeros(num_labels)
                    #pro = all_kg_proba[sub_label-1][obj_label-1]
                    rAppr = getAppr(im, rBB)    
                    rMask = np.array([getDualMask(ih, iw, sub), getDualMask(ih, iw, obj)])
                    ims.append(rAppr)
                    poses.append(rMask)
                    pros.append(pro)

            if len(ims) == 0:
                break
            ims = np.array(ims)
            poses = np.array(poses)
            subs = np.array(subs)
            objs = np.array(objs)
            pros = np.array(pros)
            num_ins = ims.shape[0]
            cursor = 0
            itr_pred = None

            # Get the predictions of predicate between subject and object
            while cursor < num_ins:
                end_batch = min(cursor + test_batch_size, num_ins)
                batch_sub = subs[cursor:end_batch,:]
                batch_obj = objs[cursor:end_batch,:]
                batch_ims = ims[cursor:end_batch,:]
                batch_proba = pros[cursor:end_batch,:]
                # batch_ims = batch_ims.transpose((0, 2, 3, 1))
                batch_poses = poses[cursor:end_batch,:]
                batch_poses = batch_poses.transpose((0, 2, 3, 1))
                #print(cursor,end_batch,batch_sub.shape,batch_poses.shape)
                feed_dict = {tf_test_sub : batch_sub, tf_test_obj: batch_obj, 
                    tf_test_ims: batch_ims, tf_test_poses: batch_poses, tf_test_proba : batch_proba, tf_keep_prob : 1.0, tf_test_batch_size: end_batch - cursor}
                predictions = session.run(test_prediction, feed_dict=feed_dict)
                #print(predictions) 
                if itr_pred is None:
                    itr_pred = predictions
                else:
                    itr_pred = np.vstack((itr_pred, predictions))
                cursor = end_batch

            cur = 0
            itr_pred = np.array(itr_pred)
            
            for objIdx in xrange(num_dets):
                if subIdx != objIdx:
                    sub_bbox = dets[subIdx, 0:4]
                    obj_bbox = dets[objIdx, 0:4]
                    for j in xrange(itr_pred.shape[1]):
                        if itr_pred[cur, j] < predic_thresh:
                            #print(itr_pred[cur, j], predic_thresh)
                            continue
                        pred[imgIdx].append([dets[subIdx, 4], itr_pred[cur, j],\
                            dets[objIdx, 4], dets[subIdx, 5], j, dets[objIdx, 5]])
                        #print(pred[imgIdx])
                        pred_bboxes[imgIdx].append([sub_bbox, obj_bbox])
                    cur += 1

        
        pred[imgIdx] = np.array(pred[imgIdx])
        pred_bboxes[imgIdx] = np.array(pred_bboxes[imgIdx])
        
        if (imgIdx % 1000 == 0):
            print("%d / %f" % (imgIdx, num_test_img))

    print("Tested.")
    print("writing file.")
    f = open('../outputs/'+output_name,'wb')
    print("Finish writing file.") 
    cp.dump([pred, pred_bboxes], f, cp.HIGHEST_PROTOCOL)
    f.close()
    session.close()
