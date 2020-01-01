# %%
# Load Data
import os
import sys
from utils.utils import load_data,datset_gen
from utils.trainer import train_with_checkpoin_tensorboard

from Models import svhn_model_simple

import tensorflow as tf
import numpy as np
import pandas as pd


root_path = "I:\\Files_ML\\Coursera\\Dl_ON_ud\\dataset"
test_path = os.path.join(root_path,"test")
train_path = os.path.join(root_path,"train")
extra_path = os.path.join(root_path,"extra")

X_test,y_test = load_data(test_path,num_only=True)
X_train,y_train = load_data(train_path,num_only=True)
X_extra,y_extra = load_data(extra_path,num_only=True)

#X_train = np.concatenate([X_train,X_extra])
#y_train = np.concatenate([y_train,y_extra])
y_train = y_train.reshape((-1,5,1))
y_test = y_test.reshape((-1,5,1))



# %%

ds_train = datset_gen(X_train,y_train,batch_size=8,buffer_size=100)
ds_test = datset_gen(X_test,y_test,batch_size=8,buffer_size=100)


# %% 
loss_cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



model = svhn_model_simple(input_shape=[None,None,3])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

# - 该模型5个位置，分别计算LOSS
model.compile(optimizer=optimizer,
              loss=loss_cce,
              metrics=['accuracy'],kernel_initializer='he_normal')

# %%
batch_size=8
train_with_checkpoin_tensorboard(model,
                X_train=ds_train,
                val_data=ds_test,
                steps_per_epoch=y_train.shape[0]//batch_size)