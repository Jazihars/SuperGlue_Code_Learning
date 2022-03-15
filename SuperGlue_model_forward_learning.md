# SuperGlue模型前向传播部分的代码学习
在这个笔记中，我将详细地分析SuperGlue模型是如何执行前向传播运算的，尤其是弄懂SuperGlue模型的训练loss是如何构造的。

## 从训练脚本中进入SuperGlue模型的前向传播代码
在SuperGlue的训练脚本`/SuperGlue-pytorch/train.py`中，首先是构造了SuperGlue的模型：
``` python
superglue = SuperGlue(config.get("superglue", {}))
```
在构造了SuperGlue的模型之后，第一次用到这个模型是在优化器的构造中：
``` python
optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
```
我们来看一下SuperGlue模型的参数长什么样子。在SuperGlue的训练脚本`/SuperGlue-pytorch/train.py`中测试下述代码（在终端中运行`python train.py`命令）：
``` python
print("----------------------开始监视代码----------------------")
print("type(superglue.parameters()): ", type(superglue.parameters()))
print("----------------------我的分割线1----------------------")
print("superglue.parameters(): ", superglue.parameters())
print("----------------------结束监视代码----------------------")
exit()
optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
```
结果为：
```
----------------------开始监视代码----------------------
type(superglue.parameters()):  <class 'generator'>
----------------------我的分割线1----------------------
superglue.parameters():  <generator object Module.parameters at 0x7f2951bab9e0>
----------------------结束监视代码----------------------
```