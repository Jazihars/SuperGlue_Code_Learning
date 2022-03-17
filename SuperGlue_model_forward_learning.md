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
这里看不出什么实质性的内容。SuperGlue模型的参数就是一个生成器对象。目前还看不出模型参数的细节。

接下来我们进入SuperGlue模型执行推理过程的核心代码：
``` python
data = superglue(pred)
```
这行代码是使用SuperGlue模型进行推理的最关键代码，位于训练过程对训练用DataLoader迭代的for循环中。我们首先来看一下输入到模型中执行推理运算的数据`pred`长什么样子。测试下述代码：
``` python
print("----------------------开始监视代码----------------------")
print("type(pred): ", type(pred))
print("----------------------我的分割线1----------------------")
for key in pred.keys():
    if type(pred[key]) == torch.Tensor:
        print(f"键 {key} 对应的值的形状为：{pred[key].shape}")
    else:
        print(f"键 {key} 对应的值为：{pred[key]}")
print("----------------------结束监视代码----------------------")
exit()
data = superglue(pred)
```
结果为：
```
----------------------开始监视代码----------------------
type(pred):  <class 'dict'>
----------------------我的分割线1----------------------
键 keypoints0 对应的值的形状为：torch.Size([1, 1, 10, 2])
键 keypoints1 对应的值的形状为：torch.Size([1, 1, 10, 2])
键 descriptors0 对应的值的形状为：torch.Size([128, 1, 10])
键 descriptors1 对应的值的形状为：torch.Size([128, 1, 10])
键 scores0 对应的值的形状为：torch.Size([10, 1])
键 scores1 对应的值的形状为：torch.Size([10, 1])
键 image0 对应的值的形状为：torch.Size([1, 1, 427, 640])
键 image1 对应的值的形状为：torch.Size([1, 1, 427, 640])
键 all_matches 对应的值的形状为：torch.Size([2, 1, 20])
键 file_name 对应的值为：['/data/zitong.yin/coco2014/train2014/COCO_train2014_000000287870.jpg']
----------------------结束监视代码----------------------
```
由此我们就明白了：**SuperGlue网络执行推理运算的时候，是把输入数据整合成了一个字典，然后输入到SuperGlue网络里的。**

我们接下来看一下SuperGlue网络执行推理运算之后的结果。测试下述代码：
``` python
data = superglue(pred)
print("----------------------开始监视代码----------------------")
print("type(data): ", type(data))
print("----------------------我的分割线1----------------------")
for key in data.keys():
    if type(data[key]) == torch.Tensor:
        print(f"键 {key} 对应的值的形状为：{data[key].shape}")
    else:
        print(f"键 {key} 对应的值为：{data[key]}")
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(data):  <class 'dict'>
----------------------我的分割线1----------------------
键 matches0 对应的值的形状为：torch.Size([10])
键 matches1 对应的值的形状为：torch.Size([10])
键 matching_scores0 对应的值的形状为：torch.Size([10])
键 matching_scores1 对应的值的形状为：torch.Size([10])
键 loss 对应的值的形状为：torch.Size([1])
键 skip_train 对应的值为：False
----------------------结束监视代码----------------------
```
由此就明白了：**SuperGlue网络推理运算之后，得到的结果也是一个字典。这个字典里包含一个名为loss的网络损失。**

接下来我们进入SuperGlue网络内部，看一下推理运算的细节。

## SuperGlue网络前向传播运算的代码分析
从SuperGlue网络的第三方训练脚本`/SuperGlue-pytorch/train.py`里，利用vscode的定义跳转功能，进入SuperGlue模型的代码，可以看到，SuperGlue网络的前向传播运算的完整代码如下：
``` python
def forward(self, data):
    """Run SuperGlue on a pair of keypoints and descriptors"""
    desc0, desc1 = data["descriptors0"].double(), data["descriptors1"].double()
    kpts0, kpts1 = data["keypoints0"].double(), data["keypoints1"].double()

    desc0 = desc0.transpose(0, 1)
    desc1 = desc1.transpose(0, 1)
    kpts0 = torch.reshape(kpts0, (1, -1, 2))
    kpts1 = torch.reshape(kpts1, (1, -1, 2))

    if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
        shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
        return {
            "matches0": kpts0.new_full(shape0, -1, dtype=torch.int)[0],
            "matches1": kpts1.new_full(shape1, -1, dtype=torch.int)[0],
            "matching_scores0": kpts0.new_zeros(shape0)[0],
            "matching_scores1": kpts1.new_zeros(shape1)[0],
            "skip_train": True,
        }

    file_name = data["file_name"]
    all_matches = data["all_matches"].permute(
        1, 2, 0
    )  # shape=torch.Size([1, 87, 2])

    # Keypoint normalization.
    kpts0 = normalize_keypoints(kpts0, data["image0"].shape)
    kpts1 = normalize_keypoints(kpts1, data["image1"].shape)

    # Keypoint MLP encoder.
    desc0 = desc0 + self.kenc(kpts0, torch.transpose(data["scores0"], 0, 1))
    desc1 = desc1 + self.kenc(kpts1, torch.transpose(data["scores1"], 0, 1))

    # Multi-layer Transformer network.
    desc0, desc1 = self.gnn(desc0, desc1)

    # Final MLP projection.
    mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

    # Compute matching descriptor distance.
    scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
    scores = scores / self.config["descriptor_dim"] ** 0.5

    # Run the optimal transport.
    scores = log_optimal_transport(
        scores, self.bin_score, iters=self.config["sinkhorn_iterations"]
    )

    # Get the matches with score above "match_threshold".
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > self.config["match_threshold"])
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    # check if indexed correctly
    loss = []
    for i in range(len(all_matches[0])):
        x = all_matches[0][i][0]
        y = all_matches[0][i][1]
        loss.append(-torch.log(scores[0][x][y].exp()))  # check batch size == 1 ?
    # for p0 in unmatched0:
    #     loss += -torch.log(scores[0][p0][-1])
    # for p1 in unmatched1:
    #     loss += -torch.log(scores[0][-1][p1])
    loss_mean = torch.mean(torch.stack(loss))
    loss_mean = torch.reshape(loss_mean, (1, -1))
    return {
        "matches0": indices0[0],  # use -1 for invalid match
        "matches1": indices1[0],  # use -1 for invalid match
        "matching_scores0": mscores0[0],
        "matching_scores1": mscores1[0],
        "loss": loss_mean[0],
        "skip_train": False,
    }

    # scores big value or small value means confidence? log can't take neg value
```
接下来我们逐行分析SuperGlue网络的前向传播代码。重点关注Loss函数的实现。

---
SuperGlue网络的前向传播函数forward(self, data)的第1-2行代码如下：
``` python
desc0, desc1 = data["descriptors0"].double(), data["descriptors1"].double()
kpts0, kpts1 = data["keypoints0"].double(), data["keypoints1"].double()
```
我们首先再来重新看一下这个`data`的样子。在`/SuperGlue-pytorch/models/superglue.py`里测试下述代码：
``` python
def forward(self, data):
    """Run SuperGlue on a pair of keypoints and descriptors"""
    print("----------------------开始监视代码----------------------")
    print("type(data): ", type(data))
    print("----------------------我的分割线1----------------------")
    for key in data.keys():
        if type(data[key]) == torch.Tensor:
            print(f"键 {key} 对应的值的形状为：{data[key].shape}")
        else:
            print(f"键 {key} 对应的值为：{data[key]}")
    print("----------------------结束监视代码----------------------")
    exit()
    desc0, desc1 = data["descriptors0"].double(), data["descriptors1"].double()
    kpts0, kpts1 = data["keypoints0"].double(), data["keypoints1"].double()
```
结果为：
```
----------------------开始监视代码----------------------
type(data):  <class 'dict'>
----------------------我的分割线1----------------------
键 keypoints0 对应的值的形状为：torch.Size([1, 1, 10, 2])
键 keypoints1 对应的值的形状为：torch.Size([1, 1, 10, 2])
键 descriptors0 对应的值的形状为：torch.Size([128, 1, 10])
键 descriptors1 对应的值的形状为：torch.Size([128, 1, 10])
键 scores0 对应的值的形状为：torch.Size([10, 1])
键 scores1 对应的值的形状为：torch.Size([10, 1])
键 image0 对应的值的形状为：torch.Size([1, 1, 427, 640])
键 image1 对应的值的形状为：torch.Size([1, 1, 427, 640])
键 all_matches 对应的值的形状为：torch.Size([2, 1, 17])
键 file_name 对应的值为：['/data/zitong.yin/coco2014/train2014/COCO_train2014_000000287870.jpg']
----------------------结束监视代码----------------------
```
由此可知，SuperGlue网络的前向传播函数forward(self, data)的第1-2行代码所做的最关键的事情就是：对PyTorch的torch.Tensor张量执行调用`.double()`函数。关于这个PyTorch张量自己的`.double()`函数，参考[PyTorch张量的double函数官方文档](https://pytorch.org/docs/1.8.0/tensors.html#torch.Tensor.double)可知，PyTorch张量的`.double()`函数的作用是：把PyTorch张量的数据格式转换成`torch.float64`类型的数据格式。我们在空白脚本中测试下述代码：
``` python
import torch


x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


print("----------------------开始监视代码----------------------")
print("x: ", x)
print("----------------------我的分割线1----------------------")
print("x.double(): ", x.double())
print("----------------------结束监视代码----------------------")
```
结果为：
```
----------------------开始监视代码----------------------
x:  tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
----------------------我的分割线1----------------------
x.double():  tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]], dtype=torch.float64)
----------------------结束监视代码----------------------
```
由此就明白了：PyTorch张量自己的`.double()`函数的功能是：将PyTorch张量的数据类型转换成`torch.float64`的数据类型。[PyTorch张量官方文档](https://pytorch.org/docs/1.8.0/tensors.html)在最开始的部分给出了各种可能的PyTorch张量数据类型。至于为什么SuperGlue网络要做这样的数据类型的转换，我目前也不是很清楚。之后再看。

---
SuperGlue网络的前向传播函数forward(self, data)的第3-4行代码如下：
``` python
desc0 = desc0.transpose(0, 1)
desc1 = desc1.transpose(0, 1)
```
这两行代码调用了PyTorch张量的`.transpose()`函数。参考[PyTorch官方torch.Tensor.transpose()文档](https://pytorch.org/docs/1.8.0/tensors.html#torch.Tensor.transpose)可知，PyTorch张量的`.transpose()`函数的作用就是：对一个PyTorch张量执行转置运算，交换它的两个维度。我们来测试一下下述代码：
``` python
print("----------------------开始监视代码----------------------")
print("desc0.shape: ", desc0.shape)

print("----------------------开始执行这行代码----------------------")
desc0 = desc0.transpose(0, 1)
print("----------------------结束执行这行代码----------------------")

print("desc0.shape: ", desc0.shape)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
desc0.shape:  torch.Size([128, 1, 10])
----------------------开始执行这行代码----------------------
----------------------结束执行这行代码----------------------
desc0.shape:  torch.Size([1, 128, 10])
----------------------结束监视代码----------------------
```
由此可知：对PyTorch张量执行`.transpose(0, 1)`变换，就是把前两个维度进行交换。