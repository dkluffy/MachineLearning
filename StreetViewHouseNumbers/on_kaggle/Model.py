import tensorflow as tf
from tensorflow import keras
from functools import partial

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

from tensorflow.keras.layers import Flatten,Dense,Activation,MaxPool2D,GlobalAvgPool2D,BatchNormalization
from tensorflow.keras.layers import Input,Conv2D
from tensorflow.keras import Model

def ResNet34(input):

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
    #return Model(inputs=[input],outputs=[y]) --这样会调用的时候 X = ResNet34(X_)(X_)，会屏蔽整个模型结构

    return y

def Model_body(input_shape=[64,64,3],N=5,class_num=11):
    X = Input(shape=input_shape)
    H = ResNet34(X)
    P_L = Dense(N+2,activation="softmax")(H)
    S = [ Dense(class_num,activation="softmax")(H) for _ in range(N) ]
    return Model( inputs=[X],outputs=[P_L]+S )

if __name__ == "__main__":
    model=Model_body(input_shape=[64,64,3])
    model.summary()