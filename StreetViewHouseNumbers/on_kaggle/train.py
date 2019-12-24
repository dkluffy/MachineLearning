# Load Data
import os
import sys
from utils.utils import load_data,to_df,to_one_hot
from Model import Model_body

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

root_path = "..\\dataset"
test_path = os.path.join(root_path,"test")
train_path = os.path.join(root_path,"train")
#extra_path = os.path.join(root_path,"extra")

X_test,y_test = load_data(test_path)
X_train,y_train = load_data(train_path)

df_train = to_df(X_train,y_train)
df_test = to_df(X_test,y_test)

#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

EPOCHS = 10
batch_size = 16

IMG_HEIGHT=128
IMG_WIDTH=128

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_dataframe(df_train,
                                                    directory=None,
                                                    x_col='filename',
                                                    y_col=["len","1","2","3","4","5"],
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode=None,
                                                    batch_size=batch_size)

val_data_gen = validation_image_generator.flow_from_dataframe(df_test,
                                                    directory=None,
                                                    x_col='filename',
                                                    y_col=["len","1","2","3","4","5"],
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode=None,
                                                    batch_size=batch_size)


run_logdir = "run_log"
model_save_file = "svhn_resnet32.h5"

total_train = len(X_train)
total_val = len(X_test)


#call backs for train
#
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_save_file)

loss_cce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#loss
def svhn_loss(y_true, y_pred):
    """
    y_pred = [...7],[...11]
    """
    #每个dense 都会附加LOSS，明天研究下

    loss = loss_cce(y_true[:,:1],y_pred[:,:1])
    return loss

#Model - using ResNet34
#
if len(sys.argv)>1:
    model = keras.models.load_model(sys.argv[1])
else:
    model = Model_body(input_shape=[IMG_HEIGHT,IMG_WIDTH,3])
    model.compile(optimizer='adam',
              loss=svhn_loss,
              metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[tensorboard_cb,checkpoint_cb]
)

# -- save --
model.save(model_save_file)


#-- visualize --
import matplotlib.pyplot as plt

#import matplotlib as mpl
#mpl.rcParams['figure.figsize'] = (8, 6)
#mpl.rcParams['axes.grid'] = False

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

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
