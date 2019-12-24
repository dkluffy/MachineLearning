# SVHN

## Intro

数据集：http://ufldl.stanford.edu/housenumbers/

[Paper](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf) （墙外）

## Implementation

用TENSORFLOW2 实现

### 数据处理

- utils/data_cooker.py

- 这个数据集的标签存在MATLAB的输出文件中，可以用H5PY读取

- 不得不吐槽下这个数据集，读起来非常麻烦，一层套一层的。项目中因为只用到`label`,所以只读了这个标签。用了递归`get_obj` 简单粗暴的展开了数据，能用就行 =。=

- 最后读取了 `image_path`(图片绝对路径)， `labels`([p,s1,s2,s3,s4,s5]),使用的时候合并成DATASET

### Model

- `Model.py/svhn_train`: 为了实现paper里的$\sum(log(P))$ 用了自定义模型和LOSS，用RESNET34替换了paper里的CNN部分。

### Todo

- [ ] 加入SAVE_WEIGHTS
- [ ] 加入tensorboard的LOG功能
- [ ] 尝试只使用Keras的自定义LOSS.参考: [keras-yolo3](https://github.com/qqwweee/keras-yolo3) （把LOSS放在输出，可以模型传递dummy_loss, model.add_loss等）

- [ ] 或者改变最后一层数和和标签格式，使得两个匹配，这样就不需要自定义模型了

- [ ] 训练模型：测试了以下，当前`Model.py/svhn_train`learning_rate=0.01 不到3个epoch 就出现梯度消失/爆炸(loss:nan)

- [ ] 尝试使用不同的CNN层，例如原作者的 或 VGG16
