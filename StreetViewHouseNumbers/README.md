# SVHN

## Intro

数据集：http://ufldl.stanford.edu/housenumbers/

[Paper](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf) （墙外）

## Implementation

用TENSORFLOW2 实现

### 数据处理

- utils/data_cooker.py

- 这个数据集的标签存在MATLAB的输出文件中，可以用H5PY读取

- 不得不吐槽下这个数据集，读起来非常麻烦，一层套一层的。项目中因为只用到`label`,所以只读了这个标签。用了递归`get_obj` 简单粗暴的展开了数据，能用就行 =。=凸

- 最后读取了 `image_path`(图片绝对路径)， `labels`([p,s1,s2,s3,s4,s5]),使用的时候合并成DATASET

### Model

- `Model.py/svhn_train`: 为了实现paper里的$\sum(log(P))$ 用了自定义模型和LOSS，用RESNET34替换了paper里的CNN部分。


#

### Train

- 计算资源：训练CNN最烦人的是计算资源，虽然本地机器只有GTX1050（2GB) +16GB超过KAGGLE上的配置，但是另外的瓶颈来自磁盘。KAGGLE上的训练速度几乎是我的本地机器的5X倍

- LOSS NAN : 将图片标准化；增加BN层； 减小BATCH； 使用HE等初始化可以解决

- 多个BATCH后LOSS 几乎没有变化（按优先级）

    - 减小LR 0.025 0.01 0.001 0.0001 ...

    - 改变初始化：有时随机初始化等不同的初始化方案会有意想不到的变化。好的初始化可以加快模型收敛

    - 改变模型结构，注意优化最后FEATURE MAP的输出至少应该满足`1 X 1 X num_classes`

    - 使用`SGD` `RMSprop`等优化器，`ADAM`有时候可能难以训练

    - 减小BATCH，16、32、64 等小BATCH非常适合CNN训练，但有些环境不适用

    - 减少输出：实际训练中发现多个输出配置多个LOSS会是模型收敛非常缓慢。所有在设计模型的时候应该尽量只使用最少输出，或者将模型拆分成几部分，收敛后再将模型合并，导入权值

    - LOSS函数：在多个输出的情况下，每个输出分别使用一个LOSS也导致了模型收敛很慢。`设计只使用一个LOSS（待验证，因为改变LOSS可能还要改变label的格式）`

# 

### Todo

- [X] 加入SAVE_WEIGHTS
- [X] 加入tensorboard的LOG功能
- [ ] 尝试只使用Keras的自定义LOSS.参考: [keras-yolo3](https://github.com/qqwweee/keras-yolo3) （把LOSS放在输出，可以模型传递dummy_loss, model.add_loss等）

- [X] 或者改变最后一层数和和标签格式，使得两个匹配，这样就不需要自定义模型了

- [X] 训练模型：测试了以下，当前`Model.py/svhn_train`learning_rate=0.01 不到3个epoch 就出现梯度消失/爆炸(loss:nan)

- [ ] 尝试使用不同的CNN层，例如原作者的 或 VGG16
- [ ] 用RNN替换SOFTMAX直接输出数字?：参考`keras.layers.Permute(dims)`

- [ ] 简单训练一个关于坐标与边框的回归损失函数

- [ ] 看到一个比较好的[模型](https://github.com/devinsaini/svhn/blob/master/jupyter/svhntrain.ipynb) 该模型有两个输入（IMG,digital_location）两个输出(字符长度，logP)， 这样需要把原来的标签拆成N个，一副图片对应N个标签. 也就是该作者把模型拆成两个部分一个只输出`字符长度`，一个只输出`digital_location 的字符`。看了作者的训练过程收敛非常快，对此表示很怀疑，他几乎没有什么优化，几个EPOCH就让LOSS下降到1以下，ACC达到80%+。
