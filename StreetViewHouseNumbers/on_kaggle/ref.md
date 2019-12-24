
## 实现略有不同

https://github.com/RyannnG/Capstone-Google-SVHN-Digits-Recognition/blob/master/5.%20Train%20Multi%202%20-%2064*64.ipynb

```python
# Training computation.
 def model:
     ...
     hidden4_drop = tf.nn.dropout(hidden4, keep_prob)

        logits_1 = tf.matmul(hidden4_drop, s1_w) + s1_b
        logits_2 = tf.matmul(hidden4_drop, s2_w) + s2_b
        logits_3 = tf.matmul(hidden4_drop, s3_w) + s3_b
        logits_4 = tf.matmul(hidden4_drop, s4_w) + s4_b
        logits_5 = tf.matmul(hidden4_drop, s5_w) + s5_b
        
    return [logits_1, logits_2, logits_3, logits_4, logits_5]

"""
作者没有在Model里输出长度，而是输出固定5个预测，labels [ 1,4,5,10,10]
作者用 10 表示空位置

论文作者输出的是SOFTMAX->ARGMAX， 该代码作者直接输出预测，计算sparse_softmax_cross_entropy_with_logits
这样不用计算ARGMAX
"""
logits = model(tf_train_dataset, 0.5, 0.8)
#logits.shape = [5,11]
loss_per_digit = [tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits[i],
                            tf_train_labels[:,i+1]
                        )) + beta_regul * tf.nn.l2_loss(sw[i])
                       for i in range(5)]

loss = tf.add_n(loss_per_digit)

# Optimizer.
learning_rate = tf.train.exponential_decay(0.001, global_step, 1000, 0.90, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

def prediction_softmax(dataset):
    prediction = tf.pack([
        tf.nn.softmax(model(dataset, 1.0, 1.0)[0]),
        tf.nn.softmax(model(dataset, 1.0, 1.0)[1]),
        tf.nn.softmax(model(dataset, 1.0, 1.0)[2]),
        tf.nn.softmax(model(dataset, 1.0, 1.0)[3]),
        tf.nn.softmax(model(dataset, 1.0, 1.0)[4])])
    return prediction

# Predictions for the training, validation, and test data.
train_prediction = prediction_softmax(tf_train_dataset)
valid_prediction = prediction_softmax(tf_valid_dataset)
test_prediction = prediction_softmax(tf_test_dataset)

```


from keras.utils.np_utils import to_categorical

注意：当使用categorical_crossentropy损失函数时，你的标签应为多类模式，例如如果你有10个类别，每一个样本的标签应该是一个10维的向量，该向量在对应有值的索引位置为1其余为0。

可以使用这个方法进行转换：

from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
————————————————
版权声明：本文为CSDN博主「趙大宝」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/u010412858/article/details/76842216


f your targets are one-hot encoded, use categorical_crossentropy.
Examples of one-hot encodings:
[1,0,0]
[0,1,0]
[0,0,1]

But if your targets are integers, use sparse_categorical_crossentropy.
Examples of integer encodings (for the sake of completion):
1
2
3

```python
#one-hot 用tf.keras.losses.CategoricalCrossentropy()
cce = tf.keras.losses.SparseCategoricalCrossentropy()
loss = cce(
  [0, 1,2],
  [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
print('Loss: ', loss.numpy())  # Loss: 0.3239


cce = tf.keras.losses.SparseCategoricalCrossentropy()
loss = cce(
  [0, 1],
  [[.9, .05, .05], [.5, .89, .6]])
print('Loss: ', loss.numpy())  # Loss: 0.3239
```