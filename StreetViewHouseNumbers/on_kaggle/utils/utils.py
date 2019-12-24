import collections
import pickle

import os
import numpy as np
import pandas as pd
def savepickle(fname,*args):
    with open(fname+"_pk","wb") as f:
        pickle.dump(args,f)
        
def loadpickle(fname):
    with open(fname,"rb") as f:
        obj = pickle.load(f)
        
    return obj

def reformat(x):
    """
    input: x = array([[array([5.]), 1])
    output: x = array([5,0,0,0,0])
    """
    x = list(x[0])
    #只截取5个，因为模型只输出5个
    x = x[:5]
    p = len(x)
    x = x + [0]*(5-p)
    return x

def load_data(rootdir,pk="digitStruct.mat_pk"):
    pk_path = os.path.join(rootdir,pk)
    image_names,labels = loadpickle(pk_path)
    
    labels_x_len = labels[:,-1:] 
    labels_x_len[labels_x_len>5]=6
    labels = np.apply_along_axis(reformat,1,labels)
    labels = np.concatenate((labels_x_len,labels),axis=1)
    image_names = [os.path.join(rootdir,x) for x in image_names]

    return image_names,labels

def to_df(x,y):
    xdf = pd.DataFrame(x,columns=["filename"])
    ydf = pd.DataFrame(y,columns=["len","1","2","3","4","5"])
    ydf = ydf.astype(int)
    
    return pd.concat([xdf,ydf],axis=1)

def to_one_hot(n_arr,cls_num):
    onehot = np.zeros((n_arr.shape[1],cls_num))
    onehot[np.arange(n_arr.shape[1]),n_arr]=1
    return onehot
    
if __name__ == "__main__":
    def test_load_data():
        rootdir = os.path.join("..\\dataset","test")
        i,b = load_data(rootdir)
        print(i[:5],b[:5])
    test_load_data()