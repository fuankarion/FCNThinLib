import sys
caffeRoot='/home/jcleon/Software/caffe/python'
sys.path.insert(0, caffeRoot)
import caffe
import lmdb
import glob
import numpy as np
from PIL import Image
import os


def createLabelLMDB(imgLabelPath,name):
    in_db = lmdb.open(name, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        in_idx=0
        labelImgs=glob.glob(imgLabelPath+'/*.jpg');
        labelImgs.sort()
        for filename in labelImgs: 
        
            print('open ',filename)
            im=np.array(Image.open(filename))
            im = im.reshape(im.shape[0], im.shape[1], 1)
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
            in_idx=in_idx+1
            
    in_db.close()

def createImageLMDB(imgLabelPath,name):
    in_db = lmdb.open(name, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        in_idx=0

        rimgs=glob.glob(imgLabelPath+'/*.jpg');
        rimgs.sort()
        for filename in rimgs: 
     
            print('open ',filename)
            im = np.array(Image.open(filename)) # or load whatever ndarray you need
            im = im[:,:,::-1]
            im = im.transpose((2,0,1))
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
            in_idx=in_idx+1
            
    in_db.close()

tagerPraefix='LMDBS'
createImageLMDB('/home/jcleon/Storage/disk0/weizmann_horse_db/pycaffe/train/rgb',os.path.join(tagerPraefix,'trainRGB'))
createLabelLMDB('/home/jcleon/Storage/disk0/weizmann_horse_db/pycaffe/train/label',os.path.join(tagerPraefix,'trainLabel'))

createImageLMDB('/home/jcleon/Storage/disk0/weizmann_horse_db/pycaffe/val/rgb',os.path.join(tagerPraefix,'valRGB'))
createLabelLMDB('/home/jcleon/Storage/disk0/weizmann_horse_db/pycaffe/val/label',os.path.join(tagerPraefix,'valLabel'))