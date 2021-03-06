import tensorflow as tf
from tensorflow import keras
from functools import partial
import numpy as np 
from tensorflow.keras.layers import Flatten,Dense,Activation,MaxPool2D,GlobalAvgPool2D,BatchNormalization
from tensorflow.keras.layers import Input,Conv2D,Lambda,Dropout
from tensorflow.keras import Model


DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

    
def ResNet34_only(input):

    """
    Use ResNet34 instead of the orig

    """ 
    #CNN network
    X = DefaultConv2D(64, kernel_size=7, strides=2)(input)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPool2D(pool_size=3, strides=2, padding="SAME")(X)
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        X = ResidualUnit(filters, strides=strides)(X)
        prev_filters = filters
    X = GlobalAvgPool2D()(X)
    y = Flatten()(X)

    #Outputs
    return y 

def CNN_org(input):
    X = DefaultConv2D(64, kernel_size=7, strides=2)(input)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPool2D(pool_size=3, strides=2, padding="SAME")(X)
    prev_filters = 64
    for filters in [64] * 3 + [128] * 2 + [256] * 3 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        X = DefaultConv2D(filters, kernel_size=2, strides=strides)(X)
        X = BatchNormalization()(X)
        X = Activation("relu")(X)
        X = MaxPool2D(pool_size=3, strides=strides, padding="SAME")(X)
        #X = Dropout(0.2)(X)
    y = Flatten()(X)

    #Outputs
    return y

def ResNet34(input,N=5,class_num=11):

    """
    Use ResNet34 instead of the orig

    """ 
    #CNN network
    X = DefaultConv2D(64, kernel_size=7, strides=2)(input)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPool2D(pool_size=3, strides=2, padding="SAME")(X)
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        X = ResidualUnit(filters, strides=strides)(X)
        prev_filters = filters
    X = GlobalAvgPool2D()(X)
    y = Flatten()(X)

    #
    L = Dense(N+2,activation="softmax")(y)
    S = [ Dense(class_num,activation="softmax")(y) for _ in range(N) ]

    #Outputs
    return Model(inputs=[input],outputs=[L]+S)

def svhn_model_simple(input_shape=[128,128,3],N=5,class_num=11):
    X=Input(shape=input_shape)
    y = ResNet34_only(X)
    y = Dense(20,activation="relu")(y)
    S = [ Dense(class_num,activation="softmax")(y) for _ in range(N) ]
    S = tf.stack(S,axis=1)
    return Model(inputs=X,outputs=S)



######################

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def loss(model, x, y):
    y_hat = model(x)
    #iterating over `tf.Tensor` is not allowed: AutoGraph did not convert this function.
    #loss = [loss_object(t,p) for t,p in zip(y,y_hat)]
    #loss = loss[0] + tf.reduce_mean(loss[1:])
    p_loss = tf.reduce_mean(loss_object(y[:,0],y_hat[0]))
    loss = [p_loss]
    for i in range(1,6):
        loss.append( tf.reduce_mean(loss_object(y[:,i],y_hat[i])) )
      
    return loss

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def svhn_train(input_ds,
               num_epochs=1000,
               learning_rate=0.003,
               input_shape=[224,224,3],
               CNNModel=ResNet34,
               N=5,class_num=11):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_loss_results = []
    train_accuracy_results = []
    

    train_log_string="Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}"

    X = Input(shape=input_shape)
    model = CNNModel(X,N,class_num)
    print("==========CNNModel Created!!!============")
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for x,y in input_ds.prefetch(buffer_size=AUTOTUNE).take(1):
            
            # 优化模型
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 追踪进度
            epoch_loss_avg(loss_value)  # 添加当前的 batch loss
            # 比较预测标签与真实标签
            #epoch_accuracy(y, model(x))
            print(train_log_string.format(epoch,epoch_loss_avg.result(),
                                              epoch_accuracy.result()))
            
      
        # 循环结束
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        print(train_log_string.format(epoch,epoch_loss_avg.result(),
                                              epoch_accuracy.result()))
      
        if epoch % 50 == 0:
            print(train_log_string.format(epoch,epoch_loss_avg.result(),
                                              epoch_accuracy.result()))
        #ToDo:
        #1.valid during training
        #2.callback:save(),log
        #for fn in callback:
        #    pass
    
    return model

def test_cnn(input_):
    #image_in_vision = Input(shape=(image_size[0],image_size[1],3))
    x = BatchNormalization()(input_)
    for filter in [32]*2+[64]*1+[128]*1 :
        x = DefaultConv2D(filter, strides=2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = DefaultConv2D(filter, strides=1, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = MaxPool2D(pool_size=(2,2), padding="SAME")(x)
   
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    h = BatchNormalization()(x)

    return Model(inputs=input_, outputs=h, name='vision')

################################    

if __name__ == "__main__":
    model=test_cnn(Input(shape=(54,128,3)))
    model.summary()