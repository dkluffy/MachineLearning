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
    return np.array(x)

def load_data(rootdir,pk="digitStruct.mat_pk",num_only=False):
    pk_path = os.path.join(rootdir,pk)
    image_names,labels = loadpickle(pk_path)
    
    labels_x_len = labels[:,-1:] 
    labels_x_len[labels_x_len>5]=6
    labels_x_len = labels_x_len.astype(float)
    labels_num = np.apply_along_axis(reformat,1,labels)
    labels = np.concatenate((labels_x_len,labels_num),axis=1)
    image_names = np.array([os.path.join(rootdir,x) for x in image_names])
    if num_only:
        return image_names,labels_num

    return image_names,labels

def to_df(x,y):
    xdf = pd.DataFrame(x,columns=["filename"])
    #这样有问题，应该 dataset = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1]})
    ydf = pd.DataFrame(y,columns=["len","1","2","3","4","5"])
    ydf = ydf.astype(int)
    
    return pd.concat([xdf,ydf],axis=1)

def to_one_hot(n_arr,cls_num):
    onehot = np.zeros((n_arr.shape[1],cls_num))
    onehot[np.arange(n_arr.shape[1]),n_arr]=1
    return onehot


import tensorflow as tf

IMG_SIZE=(128,128)   
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image /= 255.0  # normalize to [0,1] range
    image -= np.mean(image,keepdims=True)
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def datset_gen(images,lables,batch_size=16,buffer_size=1000):
    """
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    顺序很重要:

    在 .repeat 之后 .shuffle，会在 epoch 之间打乱数据（当有些数据出现两次的时候，其他数据还没有出现过）。

    在 .batch 之后 .shuffle，会打乱 batch 的顺序，但是不会在 batch 之间打乱数据。

    你在完全打乱中使用和数据集大小一样的 buffer_size（缓冲区大小）。较大的缓冲区大小提供更好的随机化，但使用更多的内存，直到超过数据集大小。

    在从随机缓冲区中拉取任何元素前，要先填满它。所以当你的 Dataset（数据集）启动的时候一个大的 buffer_size（缓冲区大小）可能会引起延迟。

    在随机缓冲区完全为空之前，被打乱的数据集不会报告数据集的结尾。Dataset（数据集）由 .repeat 重新启动，导致需要再次等待随机缓冲区被填满
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    path_ds = tf.data.Dataset.from_tensor_slices(images)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(lables)
    img_lab_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return img_lab_ds.shuffle(buffer_size=buffer_size).repeat().batch(batch_size)



import matplotlib.pyplot as plt

def plt_images(x,ylab=None,is_path=True,num=8):
    indices = np.arange(x.shape[0])
    n=1
    for i in np.random.choice(indices,num):
        plt.subplot(4,4,n)
        img = x[i]
        if is_path:
            img = tf.io.read_file(img)
            img = tf.image.decode_jpeg(img, channels=3)
         
        plt.imshow(img)
        #plt.axis("off")
        n+=2
        if ylab is not None:
            yl = np.array_str(ylab[i].flatten())
            xs = str(img.shape)
            plt.text(0,-10,"".join((yl,xs)),ha="left", va="bottom", size="medium",color="red")
    plt.show()

#plt_image_by_path(X_train,y_train)



# if __name__ == "__main__":
#     def test_load_data():
#         test_path = os.path.join("..\\dataset","test")
#         X_test,y_test = load_data(test_path)
#         print(X_test[:5],y_test[:5])
#         ds_test = datset_gen(X_test,y_test)
#         print(ds_test.take(1))
#     test_load_data()