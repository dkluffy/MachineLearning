#!/usr/bin/env python
# coding: utf-8

# convert mat file to pickle

import numpy as np
import os
import h5py
import pickle

h5_ref_type = h5py.Reference
array_type = np.ndarray
h5_dataset_type = h5py.Dataset
def get_obj(x,h5mat=None):
    if type(x) is h5_ref_type:
        return get_obj(h5mat[x],h5mat)
    if type(x) is h5_dataset_type:
        x = np.array(x).flatten()
        return np.array([ get_obj(i,h5mat) for i in x])
    if type(x) is array_type:
        #x = x.flatten()
        return np.array([ get_obj(i,h5mat) for i in x ])
    return x
    
a2s = lambda x: "".join([chr(i) for i in x])
func1 = lambda x: [a2s(i) for i in x]

def format_label(x):
    return [(i.flatten(),len(i)) for i in x]

def read_mat(matfile):
    #a2s = lambda x: "".join([chr(i) for i in x])
    #get_value = lambda x,f: a2s(np.array(f[x[0]]).flatten())

    with h5py.File(matfile, "r") as f:    
        filenames_ = f["/digitStruct/name"]
        filenames = get_obj(filenames_,f) 
        
        labels_ = f["/digitStruct/bbox"]
        labels_ = np.array(labels_).flatten()
        labels_ = np.array([f[lb]["label"] for lb in labels_])
        labels  = get_obj(labels_,f)

    return filenames,labels

def savepickle(fname,*args):
    with open(fname+"_pk","wb") as f:
        pickle.dump(args,f)
        
def loadpickle(fname):
    with open(fname,"rb") as f:
        obj = pickle.load(f)
        
    return obj

root_path = "/kaggle/input/street-view-house-numbers/"
train_path ="train_digitStruct.mat"
test_path =  "test_digitStruct.mat"
extra_path = "extra_digitStruct.mat"
  
for f in [train_path,test_path,extra_path]:
    img_names,labels = read_mat(root_path+f)
    img_names = np.apply_along_axis(func1,0,img_names)
    labels = np.apply_along_axis(format_label,0,labels)
    savepickle(f,img_names,labels)
