# %%
import os
from utils.utils import load_data
import pandas as pd
root_path = "..\\dataset"

train_path = os.path.join(root_path,"train")

X_train,y_train = load_data(train_path)


# %%
# import numpy as np 
# from utils.utils import load_data
# def reformat(x):
#     """
#     input: x = array([[array([5.]), 1])
#     output: x = array([5,0,0,0,0])
#     """
#     p = len(x[0])
#     x = list(x[0])
#     x = x + [0]*(5-p)
#     return x

# x = np.array( [ [[5.], 1], [[5.,6], 5] ])
# np.apply_along_axis(reformat,1,x)


# %%
X_train[:5],y_train[:5]

# %%
import numpy as np 
#y = np.expand_dims(y_train,axis=-1)
xdf = pd.DataFrame(X_train,columns=["filname"])
ydf = pd.DataFrame(y_train,columns=["len","1","2","3","4","5"])

# %%
xy = pd.concat([xdf,ydf],axis=1)
xy["class_new"] = xy["1"]+xy["2"] 
xy[:5]

# %%
def to_one_hot(n_arr,cls_num):
    onehot = np.zeros((len(n_arr),cls_num))
    onehot[np.arange(len(n_arr)),n_arr]=1
    return onehot
to_one_hot([6],7)
# %%
