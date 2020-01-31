import cPickle as cp
import numpy as np

print("read pkl...")
entity_list = []
rel_list = []
with open('object_class.txt','r') as fid:
    lines = fid.readlines()
    for line in lines:
         entity_list.append(line.strip())     
with open('predicate_class.txt','r') as fid:
    lines = fid.readlines()
    for line in lines:
         rel_list.append(line.strip())
object_cate_num = len(entity_list)
predicate_cate_num = len(rel_list)

with open('train/train_gt_file.pkl')as fid:
    train_gt_list = cp.load(fid)
with open('test/test_gt_file.pkl')as fid:
    test_gt_list = cp.load(fid)

print "initialize..."
all_relation_list = []
str_all_relation_list = []
for i in range(len(train_gt_list['gt_label'])):
    gt_label = train_gt_list['gt_label'][i]
    #gt_boxes = train_gt_list['gt_box'][i]
    for tri_label in gt_label:
        str_tri_label = str(tri_label)
        if str_tri_label not in str_all_relation_list:
            str_all_relation_list.append(str_tri_label)
            all_relation_list.append(tri_label)

aa = {}   
bb = {}
cc = {} 
for obj1 in xrange(object_cate_num):#0~100
    aa[obj1] = {}
    bb[obj1] = {}
    cc[obj1] = {}
    for obj2 in xrange(object_cate_num):#0~70
        aa[obj1][obj2] = {}
        bb[obj1][obj2] = {}
        cc[obj1][obj2] = {}
        for pre in xrange(predicate_cate_num):
            aa[obj1][obj2][pre] = 0
            bb[obj1][obj2][pre] = np.array([0,0,0,0])
            cc[obj1][obj2][pre] = 0

print("compute mean box...")
for i in range(len(train_gt_list['gt_label'])):
    gt_label = train_gt_list['gt_label'][i]
    gt_boxes = train_gt_list['gt_box'][i]
    for tri_label,tri_box in zip(gt_label,gt_boxes):
        [obj1,pre,obj2] = tri_label
        obj1 = obj1 - 1
        obj2 = obj2 - 1
        [obox,sbox] = tri_box
        w = float(obox[2]-obox[0])
        w_ = float(sbox[2]-sbox[0])
        h = float(obox[3]-obox[1])
        h_ = float(sbox[3]-sbox[1])
        tx = (obox[0]-sbox[0])/float(w_)
        ty = (obox[1]-sbox[1])/float(h_)
        tw = np.log(float(w/w_))
        th = np.log(float(h/h_))
        rbox = [tx,ty,tw,th]
        #print obj1,obj2,pre
        bb[obj1][obj2][pre] = np.add(bb[obj1][obj2][pre],rbox)
        cc[obj1][obj2][pre] += 1.0

        
for obj1 in xrange(object_cate_num):
    for obj2 in xrange(object_cate_num):
        for pre in xrange(predicate_cate_num):
            count = float(cc[obj1][obj2][pre])
            total = bb[obj1][obj2][pre]
            if count>0.0:
                bb[obj1][obj2][pre] = total/count
            
            
print("compute probability...")
for tri in all_relation_list:
    [obj1,pre,obj2] = tri
    obj1=obj1-1
    obj2=obj2-1
    if obj1 in aa and obj2 in aa[obj1] and pre in aa[obj1][obj2]:
        aa[obj1][obj2][pre] += 1
    else:
        print("ERROR!")
        print(tri)
    
for obj1 in xrange(object_cate_num):
    for obj2 in xrange(object_cate_num):
        nums  = 0
        for pre in aa[obj1][obj2]:
            nums+= aa[obj1][obj2][pre]
        if nums == 0:
            continue
        for pre in aa[obj1][obj2]:
            aa[obj1][obj2][pre] =  aa[obj1][obj2][pre]*1.0/(nums*1.0)
print("compute done")
print("write pkl...")
with  open('/home/lab/LN/new_work/outputs/CKG_vrd_pro.pkl', 'wb')as fid:
    cp.dump(aa, fid, cp.HIGHEST_PROTOCOL)
with  open('/home/lab/LN/new_work/outputs/CKG_vrd_box.pkl', 'wb')as fid:
    cp.dump(bb, fid, cp.HIGHEST_PROTOCOL)
with  open('/home/lab/LN/new_work/outputs/CKG_vrd_numbox.pkl', 'wb')as fid:
    cp.dump(cc, fid, cp.HIGHEST_PROTOCOL)

print("Finish writing file.") 