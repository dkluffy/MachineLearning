# %%
# Load Data
import os
import sys
from utils.utils import load_data,datset_gen

from Model import svhn_train

import tensorflow as tf
import numpy as np
import pandas as pd


root_path = "..\\dataset"
test_path = os.path.join(root_path,"test")
train_path = os.path.join(root_path,"train")
#extra_path = os.path.join(root_path,"extra")

#X_test,y_test = load_data(test_path)
X_train,y_train = load_data(train_path)


# %%
#ds_test = datset_gen(X_test,y_test)
ds_train = datset_gen(X_train,y_train,batch_size=8,buffer_size=100)

# %%
tf.keras.backend.clear_session()
#with tf.device('/CPU:0'):
model = svhn_train(ds_train)


