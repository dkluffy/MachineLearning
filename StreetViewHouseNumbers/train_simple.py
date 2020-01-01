# %%
# Load Data
import os
import sys
from utils.utils import load_data,datset_gen,savepickle
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
batch_size=32

ds_train = datset_gen(X_train,y_train,batch_size=batch_size,buffer_size=100)
ds_test = datset_gen(X_test,y_test,batch_size=batch_size,buffer_size=100)


# %% 
loss_cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



model = svhn_model_simple(input_shape=[None,None,3])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01,clipvalue=1.)

# - 该模型5个位置，分别计算LOSS
model.compile(optimizer=optimizer,
              loss=loss_cce,
              metrics=['accuracy'],kernel_initializer='he_normal')

# %%
num_epoches=1
with tf.device("/GPU:0"):
    history = model.fit_generator(ds_train,
                epochs=num_epoches,
                validation_data=ds_test,
                validation_steps=100,
                steps_per_epoch=y_train.shape[0]//batch_size)


model.save("model_save_file.h5")

#-- visualize --
import matplotlib.pyplot as plt

#import matplotlib as mpl
#mpl.rcParams['figure.figsize'] = (8, 6)
#mpl.rcParams['axes.grid'] = False

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(num_epoches)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()