# 一些PyTorch的基本用法

## 网络结构是如何表示的？
在空白脚本里测试如下的代码：
``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
```
结果为：
```
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```
由此可知：`print(net)`打印出来的是网络实例`net`的类`Net`定义中的初始化函数`def __init__(self):`的内容。

## 网络参数是如何存储的？
网络中的可学习参数是通过`net.state_dict()`这个有序字典来存储的。我们在空白脚本中测试如下的代码：
``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


print("----------------------开始监视代码----------------------")
print("type(net.state_dict())：", type(net.state_dict()))
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print("----------------------我的分割线1----------------------")
print("type(optimizer.state_dict())：", type(optimizer.state_dict()))
# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

print("----------------------结束监视代码----------------------")
```
结果为：
```
----------------------开始监视代码----------------------
type(net.state_dict())： <class 'collections.OrderedDict'>
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias       torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias       torch.Size([16])
fc1.weight       torch.Size([120, 400])
fc1.bias         torch.Size([120])
fc2.weight       torch.Size([84, 120])
fc2.bias         torch.Size([84])
fc3.weight       torch.Size([10, 84])
fc3.bias         torch.Size([10])
----------------------我的分割线1----------------------
type(optimizer.state_dict())： <class 'dict'>
Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]
----------------------结束监视代码----------------------
```
我们可以看到，`net.state_dict()`的类型是一个有序字典。这个字典中存储了网络`net`的所有可学习参数。字典的键就是每个可学习层的参数名称（在上面的例子中，就是`conv1.weight`，`conv1.bias`，`conv2.weight`，`conv2.bias `，`fc1.weight`，`fc1.bias`，`fc2.weight`，`fc2.bias`，`fc3.weight`，`fc3.bias`）。字典的值就是对应的参数（这些参数都是一些PyTorch张量）。
类似地，优化器的参数也是一个字典。这个字典里面保存了`state`和`param_groups`这两个键。

以上内容参考[PyTorch状态字典教程](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)