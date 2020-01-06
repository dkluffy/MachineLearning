# CAD图纸标注

这个问题挺有意思，看上去挺简单，但是用深度学习去解，貌似也不简单。
还没时间取做，先记下来。

## 问题

给定一些中CAD图纸中提取的线段信息（起点和终点，线段类型），标注出大矩形（包含子图形）、小矩形、三角形等

## 我的解答

我的大致解题思路：

### 采用端到端的模型

### 输入

图纸（把点线信息绘制在一定大小的图片中）+ 线段起止坐标 （即原输入中线段的信息 （0，2），（2，2））

### 输出

[类1（大矩形，小矩形，三角形，无）线段起止坐标1 ；类2（大矩形，小矩形，三角形，无），线段起止坐标2]

 因为 一条线段可以是多个矩形/三角形的边，所以最少要输出两个（最大的两个），输出的坐标就是和输入坐标构成了图形的平行线段，如果是三角形线段的起始坐标等于终止坐标。

### 模型结构

采用CNN+FC层构建一个类别识别模型，从我构建的输出类型可以看出，我把问题归为分类问题

### 构建训练数据

可以代码构建一个较大的数据集，比如 在固定大小的画布上随机取两点画图 ，标签可以在作图过程中保留下来。并可以通过旋转/缩放图形（不改变画布大小，变换后标签也要相应改变）来获得更多的样本。

### 训练LOSS

LOSS分为两部分之和：一个是类别，另一个线段。线段部分可以设计成 （预测起止点到标签起止点距离距离之和，模型的目的就是通过拟合把这两对点的距离缩小到最小）

### 预测输出

把需要预测的所有线段信息 逐个和 图纸一起输入到模型；把所有输出汇总到一起，去掉重复的就可以标注出矩形、三角形。