# SuperGlue模型部分的代码学习
在这个笔记中，我将逐行阅读[SuperGlue模型部分的代码](https://github.com/HeatherJiaZG/SuperGlue-pytorch/blob/master/models/superglue.py)，弄懂SuperGlue模型究竟是怎么实现的。

## 从训练脚本中进入模型部分的代码——模型构造
在[SuperGlue的第三方训练脚本](https://github.com/HeatherJiaZG/SuperGlue-pytorch/blob/master/train.py)`/SuperGlue-pytorch/train.py`中，可以看到这样的几行代码（注意：我的代码经过了Python第三方包black的强制格式化，所以与Github上原始作者的代码格式不同。）：
``` python
config = {
    "superpoint": {
        "nms_radius": opt.nms_radius,
        "keypoint_threshold": opt.keypoint_threshold,
        "max_keypoints": opt.max_keypoints,
    },
    "superglue": {
        "weights": opt.superglue,
        "sinkhorn_iterations": opt.sinkhorn_iterations,
        "match_threshold": opt.match_threshold,
    },
}

train_set = SparseDataset(opt.train_path, opt.max_keypoints)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=False
)

superglue = SuperGlue(config.get("superglue", {}))
```
可以看到，初始化模型的最关键的代码是
``` python
superglue = SuperGlue(config.get("superglue", {}))
```
这一行代码。初始化SuperGlue模型需要的所有参数都是由一个名为`config`的字典传递给SuperGlue模型的构造函数的。我们先来对这一行初始化模型的代码进行一些简单的测试。首先，我们来看看初始化模型时是否会打印出一些内容。测试下述代码：
``` python
print("----------------------开始监视代码----------------------")
superglue = SuperGlue(config.get("superglue", {}))
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
----------------------结束监视代码----------------------
```
可以看到，在初始化模型的过程中，没有任何打印输出的内容。
我们再来看看初始化模型所用到的参数。测试下述代码：
``` python
print("----------------------开始监视代码----------------------")
print("type(config): ", type(config))
print("----------------------我的分割线1----------------------")
print("config: ", config)
print("----------------------我的分割线2----------------------")
print('type(config.get("superglue", {})): ', type(config.get("superglue", {})))
print("----------------------我的分割线3----------------------")
print('config.get("superglue", {}): ', config.get("superglue", {}))
print("----------------------结束监视代码----------------------")
exit()
superglue = SuperGlue(config.get("superglue", {}))
```
结果为：
```
----------------------开始监视代码----------------------
type(config):  <class 'dict'>
----------------------我的分割线1----------------------
config:  {'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 10}, 'superglue': {'weights': 'indoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}}
----------------------我的分割线2----------------------
type(config.get("superglue", {})):  <class 'dict'>
----------------------我的分割线3----------------------
config.get("superglue", {}):  {'weights': 'indoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
----------------------结束监视代码----------------------
```
由此我们就明白了：**SuperGlue的模型，是用一个参数字典来初始化的。这个参数字典里包含了三个参数：`'weights'`，`'sinkhorn_iterations'`和`'match_threshold'`。因此，SuperGlue模型的初始化必须指定这三个参数的值。**
另外，有一点值得注意：**在SuperGlue的训练脚本`/SuperGlue-pytorch/train.py`中，字典`config`仅出现了两次。第一次是构造`config`字典的时候，第二次是初始化SuperGlue模型的时候。由此可知：`config`字典是专门为了初始化SuperGlue模型而构造的，除此以外，这个`config`字典并没有任何其他的用途。**

接下来我们进入SuperGlue模型的内部，来详细地研究一下SuperGlue模型的实现细节。

## SuperGlue模型构造部分的代码逐行分析
从SuperGlue的训练脚本`/SuperGlue-pytorch/train.py`中的模型初始化代码
``` python
superglue = SuperGlue(config.get("superglue", {}))
```
处，利用vscode自带的定义跳转功能（Ctrl+鼠标左键单击），进入[SuperGlue模型的定义](https://github.com/HeatherJiaZG/SuperGlue-pytorch/blob/master/models/superglue.py#L180)`/SuperGlue-pytorch/models/superglue.py`，可以看到，第三方训练脚本用到的SuperGlue模型类的完整代码如下：
``` python
class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """

    default_config = {
        "descriptor_dim": 128,
        "weights": "indoor",
        "keypoint_encoder": [32, 64, 128],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config["descriptor_dim"], self.config["keypoint_encoder"]
        )

        self.gnn = AttentionalGNN(
            self.config["descriptor_dim"], self.config["GNN_layers"]
        )

        self.final_proj = nn.Conv1d(
            self.config["descriptor_dim"],
            self.config["descriptor_dim"],
            kernel_size=1,
            bias=True,
        )

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        # self.load_state_dict(torch.load(path))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config['weights']))

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
接下来我们来**逐行分析**第三方训练脚本使用的SuperGlue类的代码。特别是，之后要弄清楚，**训练的Loss函数是如何实现的**。同时，我们也要**对比分析**原始SuperGlue论文作者实现的SuperGlue类的代码和第三方训练脚本使用的SuperGlue类的代码有什么不同之处，这样我们就能更清楚地明白第三方训练脚本是如何复现SuperGlue的训练代码的。

### SuperGlue模型类的注释以及默认初始化参数
在SuperGlue模型类的初始化函数__init__(self, config)之前，有一段SuperGlue模型类的注释和默认的初始化参数：
``` python
"""SuperGlue feature matching middle-end

Given two sets of keypoints and locations, we determine the
correspondences by:
    1. Keypoint Encoding (normalization + visual feature and location fusion)
    2. Graph Neural Network with multiple self and cross-attention layers
    3. Final projection layer
    4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
    5. Thresholding matrix based on mutual exclusivity and a match_threshold

The correspondence ids use -1 to indicate non-matching points.

Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

"""

default_config = {
    "descriptor_dim": 128,
    "weights": "indoor",
    "keypoint_encoder": [32, 64, 128],
    "GNN_layers": ["self", "cross"] * 9,
    "sinkhorn_iterations": 100,
    "match_threshold": 0.2,
}
```
我们看到，注释中的内容是：
```
SuperGlue特征匹配中端
给定两组关键点和相应的关键点位置，我们通过以下方式确定对应关系：
1. 关键点编码（归一化+视觉特征和位置融合）
2. 具有多个自注意力机制层和交叉注意力机制层的图神经网络
3. 最后的投影层
4. 最优传输层（一种可微分的匈牙利匹配算法）
5. 基于互斥性和一个匹配阈值的阈值矩阵

在关键点对应的id中使用-1来表示非匹配点。

Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. SuperGlue: 用图神经网络学习特征匹配. CVPR2020. https://arxiv.org/abs/1911.11763
```
关于SuperGlue模型的更多细节信息，可以参考论文中的相应描述。
`default_config`是一个默认的初始化参数字典，之后用到的时候再来分析。


### SuperGlue模型类的初始化函数__init__(self, config)
第三方训练脚本使用的SuperGlue模型类的初始化函数__init__(self, config)的代码如下（注意：可以对照一下[原始SuperGlue论文作者的实现](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py#L206)）：
``` python
def __init__(self, config):
    super().__init__()
    self.config = {**self.default_config, **config}

    self.kenc = KeypointEncoder(
        self.config["descriptor_dim"], self.config["keypoint_encoder"]
    )

    self.gnn = AttentionalGNN(
        self.config["descriptor_dim"], self.config["GNN_layers"]
    )

    self.final_proj = nn.Conv1d(
        self.config["descriptor_dim"],
        self.config["descriptor_dim"],
        kernel_size=1,
        bias=True,
    )

    bin_score = torch.nn.Parameter(torch.tensor(1.0))
    self.register_parameter("bin_score", bin_score)

    # assert self.config['weights'] in ['indoor', 'outdoor']
    # path = Path(__file__).parent
    # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
    # self.load_state_dict(torch.load(path))
    # print('Loaded SuperGlue model (\"{}\" weights)'.format(
    #     self.config['weights']))
```
可以看到，[原始SuperGlue论文作者的实现](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py#L206)如下：
``` python
def __init__(self, config):
    super().__init__()
    self.config = {**self.default_config, **config}

    self.kenc = KeypointEncoder(
        self.config["descriptor_dim"], self.config["keypoint_encoder"]
    )

    self.gnn = AttentionalGNN(
        feature_dim=self.config["descriptor_dim"],
        layer_names=self.config["GNN_layers"],
    )

    self.final_proj = nn.Conv1d(
        self.config["descriptor_dim"],
        self.config["descriptor_dim"],
        kernel_size=1,
        bias=True,
    )

    bin_score = torch.nn.Parameter(torch.tensor(1.0))
    self.register_parameter("bin_score", bin_score)

    assert self.config["weights"] in ["indoor", "outdoor"]
    path = Path(__file__).parent
    path = path / "weights/superglue_{}.pth".format(self.config["weights"])
    self.load_state_dict(torch.load(str(path)))
    print('Loaded SuperGlue model ("{}" weights)'.format(self.config["weights"]))
```
可以看到，第三方复现的训练脚本中使用的SuperGlue模型和原始SuperGlue论文作者使用的SuperGlue模型还是有所不同的。我们以第三方训练脚本用到的模型为基准来进行逐行的代码分析，同时我们也会关注第三方复现的训练脚本中使用的SuperGlue模型和原始SuperGlue论文作者使用的SuperGlue模型中的不同点。

---
第三方训练脚本使用的SuperGlue模型类的初始化函数__init__(self, config)的第1行代码如下：
``` python
super().__init__()
```
参考stackoverflow上的[What does 'super' do in Python? - difference between super().__init__() and explicit superclass __init__()](https://stackoverflow.com/questions/222877/what-does-super-do-in-python-difference-between-super-init-and-expl) 和 [Understanding Python super() with __init__() methods [duplicate]](https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods)，以及[Python3.8 super()函数官方文档](https://docs.python.org/3.8/library/functions.html#super)，可以明白：**Python3.8中`super().__init__()`函数的使用可以使别人更容易调用我所写的类。** 所以，这句话其实是一种编写代码的习惯的体现。

---
第三方训练脚本使用的SuperGlue模型类的初始化函数__init__(self, config)的第2行代码如下：
``` python
self.config = {**self.default_config, **config}
```
这句话把默认的参数字典和传入SuperGlue模型类的构造函数__init__(self, config)的参数字典合并为一个字典。我们来看看这个字典最终长什么样子。测试下述代码（注意：执行代码时，还是在/SuperGlue-pytorch目录下执行`python train.py`命令）：
``` python
print("----------------------开始监视代码----------------------")
print("self.default_config: ", self.default_config)
print("----------------------我的分割线1----------------------")
print("config: ", config)

print("----------------------开始执行这行代码----------------------")
self.config = {**self.default_config, **config}
print("----------------------结束执行这行代码----------------------")

print("self.config: ", self.config)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
self.default_config:  {'descriptor_dim': 128, 'weights': 'indoor', 'keypoint_encoder': [32, 64, 128], 'GNN_layers': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'], 'sinkhorn_iterations': 100, 'match_threshold': 0.2}
----------------------我的分割线1----------------------
config:  {'weights': 'indoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
----------------------开始执行这行代码----------------------
----------------------结束执行这行代码----------------------
self.config:  {'descriptor_dim': 128, 'weights': 'indoor', 'keypoint_encoder': [32, 64, 128], 'GNN_layers': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'], 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
----------------------结束监视代码----------------------
```
可以看到，**`self.config = {**self.default_config, **config}`这行代码的作用就是：把`self.default_config`和`config`这两个字典的内容合并到`self.config`这个字典里面。其中，如果两个字典有相同的键，值的内容就会以后面的这个`config`字典为准。**

---
第三方训练脚本使用的SuperGlue模型类的初始化函数__init__(self, config)的第3行代码如下：
``` python
self.kenc = KeypointEncoder(
    self.config["descriptor_dim"], self.config["keypoint_encoder"]
)
```
从这一行代码开始，第三方训练脚本使用的SuperGlue模型类的初始化函数__init__(self, config)就开始构造SuperGlue网络的各个模块了。这一行代码构造了关键点编码器模块。我们来看一下关键点编码器类的完整定义（这个类的定义位于`/SuperGlue-pytorch/models/superglue.py`脚本里）：
``` python
class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))
```
可以看到，SuperGlue的关键点编码器`KeypointEncoder`模块的构造函数接收两个输入。第一个输入`feature_dim`从名称来推断，应该是一个整数，表示特征维度。在本次训练中，实际传入的值是`128`（参见之前的打印输出`'descriptor_dim': 128`）。第二个输入`layers`从名称来推断，应该表示的是关键点编码器的层。在本次训练中，实际传入的值是`[32, 64, 128]`（参见之前的打印输出`'keypoint_encoder': [32, 64, 128]`）。

经过与[SuperGlue官方模型的class KeypointEncoder(nn.Module):代码](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py#L75)的对比发现，第三方实现的`class KeypointEncoder(nn.Module)`类与官方实现的`class KeypointEncoder(nn.Module)`类完全一样。在这个类中，没有代码的更改。

我们先来看看`class KeypointEncoder(nn.Module)`类的构造函数做了什么事情。先来看看`KeypointEncoder`类的构造函数中传入的列表长什么样子。测试一下下述代码：
``` python
class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()

        print("----------------------开始监视代码----------------------")
        print(
            "type([3] + layers + [feature_dim]): ", type([3] + layers + [feature_dim])
        )
        print("----------------------我的分割线1----------------------")
        print("[3] + layers + [feature_dim]: ", [3] + layers + [feature_dim])
        print("----------------------结束监视代码----------------------")
        exit()

        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))
```
结果为：
```
----------------------开始监视代码----------------------
type([3] + layers + [feature_dim]):  <class 'list'>
----------------------我的分割线1----------------------
[3] + layers + [feature_dim]:  [3, 32, 64, 128, 128]
----------------------结束监视代码----------------------
```
也就是说，`KeypointEncoder`类的构造函数使用一个列表构造了一个多层感知机。我们来看看这个多层感知机长什么样子。`self.encoder = MLP([3] + layers + [feature_dim])`这行代码中的MLP类的完整代码是（位于`/SuperGlue-pytorch/models/superglue.py`里）：
``` python
def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
```
与官方SuperGlue模型对比后发现，唯一的区别在注释掉的`# layers.append(nn.BatchNorm1d(channels[i]))`这一行代码上。官方SuperGlue模型使用的是注释掉的这一行代码，而第三方实现对这一行代码进行了修改，把`BatchNorm1d`换成了`InstanceNorm1d`。对于这个多层感知机的代码，没有什么特别的难点，就是构造多层感知机的常规操作。我们只需要看看这个多层感知机的结构就可以了。测试下述代码：
``` python
def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    print("----------------------开始监视代码----------------------")
    for temp in layers:
        print(temp)
    print("----------------------结束监视代码----------------------")
    exit()
    return nn.Sequential(*layers)
```
结果为：
```
----------------------开始监视代码----------------------
Conv1d(3, 32, kernel_size=(1,), stride=(1,))
InstanceNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
ReLU()
Conv1d(32, 64, kernel_size=(1,), stride=(1,))
InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
ReLU()
Conv1d(64, 128, kernel_size=(1,), stride=(1,))
InstanceNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
ReLU()
Conv1d(128, 128, kernel_size=(1,), stride=(1,))
----------------------结束监视代码----------------------
```
由此，这个多层感知机的结构就一目了然了。我们再来看看最终这个构造好了的多层感知机对象是什么类型的。测试下述代码：
``` python
class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()

        self.encoder = MLP([3] + layers + [feature_dim])
        print("----------------------开始监视代码----------------------")
        print("type(self.encoder): ", type(self.encoder))
        print("----------------------我的分割线1----------------------")
        print("self.encoder: ", self.encoder)
        print("----------------------结束监视代码----------------------")
        exit()

        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))
```
结果为：
```
----------------------开始监视代码----------------------
type(self.encoder):  <class 'torch.nn.modules.container.Sequential'>
----------------------我的分割线1----------------------
self.encoder:  Sequential(
  (0): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
  (1): InstanceNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (2): ReLU()
  (3): Conv1d(32, 64, kernel_size=(1,), stride=(1,))
  (4): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (5): ReLU()
  (6): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
  (7): InstanceNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (8): ReLU()
  (9): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
)
----------------------结束监视代码----------------------
```
也就是说，**关键点编码器构造的多层感知机是一个`<class 'torch.nn.modules.container.Sequential'>`类型的对象。这个多层感知机是把各个层封装到一个PyTorch官方的torch.nn.modules.container.Sequential容器对象里的。**
**注意：PyTorch模型的构造方法，是一些标准的用法。以后我应该从各种优秀的开源代码中学习更多的PyTorch模型构造方法。一定要多从优秀的开源代码中学习人家的写法。**

对于关键点编码器的构造函数中的最后一行代码`nn.init.constant_(self.encoder[-1].bias, 0.0)`，我们先来看看这行代码接受的输入。测试一下下述代码：
``` python
class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])

        print("----------------------开始监视代码----------------------")
        print("type(self.encoder[-1]): ", type(self.encoder[-1]))
        print("----------------------我的分割线1----------------------")
        print("self.encoder[-1]: ", self.encoder[-1])
        print("----------------------我的分割线2----------------------")
        print("type(self.encoder[-1].bias): ", type(self.encoder[-1].bias))
        print("----------------------我的分割线3----------------------")
        print("self.encoder[-1].bias.shape: ", self.encoder[-1].bias.shape)
        print("----------------------我的分割线4----------------------")
        print("self.encoder[-1].bias: ", self.encoder[-1].bias)
        print("----------------------结束监视代码----------------------")
        exit()
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))
```
结果为：
```
----------------------开始监视代码----------------------
type(self.encoder[-1]):  <class 'torch.nn.modules.conv.Conv1d'>
----------------------我的分割线1----------------------
self.encoder[-1]:  Conv1d(128, 128, kernel_size=(1,), stride=(1,))
----------------------我的分割线2----------------------
type(self.encoder[-1].bias):  <class 'torch.nn.parameter.Parameter'>
----------------------我的分割线3----------------------
self.encoder[-1].bias.shape:  torch.Size([128])
----------------------我的分割线4----------------------
self.encoder[-1].bias:  Parameter containing:
tensor([-0.0591, -0.0655,  0.0222,  0.0279,  0.0781,  0.0686,  0.0114,  0.0864,
        -0.0775,  0.0293, -0.0544, -0.0693,  0.0281,  0.0615, -0.0345,  0.0415,
        -0.0357,  0.0082, -0.0797,  0.0467,  0.0745,  0.0516, -0.0690, -0.0045,
        -0.0531, -0.0249, -0.0022,  0.0104, -0.0587,  0.0761, -0.0828, -0.0761,
        -0.0333,  0.0600,  0.0014, -0.0308,  0.0073, -0.0569, -0.0606,  0.0245,
        -0.0344,  0.0499, -0.0349, -0.0115,  0.0732,  0.0021,  0.0136, -0.0276,
         0.0142,  0.0752, -0.0819, -0.0025, -0.0039, -0.0488,  0.0792,  0.0543,
         0.0503, -0.0029,  0.0130,  0.0230, -0.0124,  0.0015, -0.0147, -0.0770,
        -0.0381, -0.0581, -0.0088, -0.0647, -0.0734,  0.0272, -0.0413,  0.0119,
         0.0235,  0.0637, -0.0675, -0.0612, -0.0053, -0.0862,  0.0288,  0.0509,
        -0.0553, -0.0026, -0.0070, -0.0372,  0.0702, -0.0164,  0.0493,  0.0419,
        -0.0856,  0.0406,  0.0706, -0.0471, -0.0618, -0.0161,  0.0274, -0.0190,
        -0.0314, -0.0135,  0.0144, -0.0584,  0.0046, -0.0372, -0.0555, -0.0273,
        -0.0446, -0.0420,  0.0170,  0.0041,  0.0579, -0.0584, -0.0369,  0.0492,
         0.0435,  0.0286,  0.0281,  0.0033,  0.0798,  0.0518,  0.0857, -0.0634,
        -0.0258,  0.0142,  0.0172,  0.0232,  0.0046,  0.0481,  0.0345, -0.0107],
       requires_grad=True)
----------------------结束监视代码----------------------
```
由此我们学到了：**对于PyTorch官方的`<class 'torch.nn.modules.container.Sequential'>`类型的模型结构，我们可以通过数组索引的方式来访问里面的每一个层。** 比如这里看到的，`self.encoder`是长这个样子的结构：
```
self.encoder:  Sequential(
  (0): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
  (1): InstanceNorm1d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (2): ReLU()
  (3): Conv1d(32, 64, kernel_size=(1,), stride=(1,))
  (4): InstanceNorm1d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (5): ReLU()
  (6): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
  (7): InstanceNorm1d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (8): ReLU()
  (9): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
)
```
self.encoder[-1]就是这个网络的最后一层，长这个样子：
```
self.encoder[-1]:  Conv1d(128, 128, kernel_size=(1,), stride=(1,))
```
self.encoder[-1].bias就是这个最后一层的偏置，长这个样子：
```
type(self.encoder[-1].bias):  <class 'torch.nn.parameter.Parameter'>
self.encoder[-1].bias.shape:  torch.Size([128])
```
**对于类似这样的PyTorch官方用法，一定要之后多看优秀的开源代码，逐渐熟悉和积累这些用法。**
关于`nn.init.constant_`的用法，参见[torch.nn.init.constant_(tensor, val)官方文档](https://pytorch.org/docs/1.8.0/nn.init.html#torch.nn.init.constant_)可知，`torch.nn.init.constant_(tensor, val)`函数的作用是：用给定的值`val`来填充一个给定的PyTorch张量。我们来看看这个`torch.nn.init.constant_(tensor, val)`函数的执行效果。测试下述代码：
``` python
class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])

        print("----------------------开始监视代码----------------------")
        print("self.encoder[-1].bias.shape: ", self.encoder[-1].bias.shape)
        print("----------------------我的分割线1----------------------")
        print("self.encoder[-1].bias: ", self.encoder[-1].bias)
        print("----------------------开始执行这行代码----------------------")

        nn.init.constant_(self.encoder[-1].bias, 0.0)

        print("----------------------结束执行这行代码----------------------")
        print("self.encoder[-1].bias.shape: ", self.encoder[-1].bias.shape)
        print("----------------------我的分割线2----------------------")
        print("self.encoder[-1].bias: ", self.encoder[-1].bias)
        print("----------------------结束监视代码----------------------")
        exit()

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))
```
结果为：
```
----------------------开始监视代码----------------------
self.encoder[-1].bias.shape:  torch.Size([128])
----------------------我的分割线1----------------------
self.encoder[-1].bias:  Parameter containing:
tensor([ 0.0805, -0.0722, -0.0326,  0.0694,  0.0372,  0.0200,  0.0475,  0.0758,
        -0.0446,  0.0182, -0.0410,  0.0441,  0.0542, -0.0691, -0.0274, -0.0563,
         0.0345,  0.0773,  0.0312, -0.0380,  0.0662, -0.0425,  0.0333,  0.0199,
         0.0876,  0.0610,  0.0381,  0.0100, -0.0225,  0.0839,  0.0513,  0.0743,
        -0.0631, -0.0577, -0.0754, -0.0680, -0.0463, -0.0467, -0.0809, -0.0845,
        -0.0004, -0.0380, -0.0017,  0.0847,  0.0584, -0.0241,  0.0489, -0.0427,
         0.0806, -0.0480,  0.0674, -0.0574, -0.0672, -0.0092, -0.0186, -0.0164,
        -0.0866,  0.0361,  0.0045,  0.0760, -0.0214,  0.0707,  0.0400,  0.0605,
         0.0259, -0.0211, -0.0011,  0.0588,  0.0033, -0.0072, -0.0475, -0.0436,
        -0.0615,  0.0440,  0.0564, -0.0105,  0.0430, -0.0397,  0.0611, -0.0669,
        -0.0360,  0.0507,  0.0870,  0.0136,  0.0408,  0.0573, -0.0797,  0.0333,
        -0.0769,  0.0482,  0.0428,  0.0876, -0.0671, -0.0130, -0.0715, -0.0033,
        -0.0634, -0.0046, -0.0529, -0.0796, -0.0326, -0.0204,  0.0215, -0.0281,
         0.0558,  0.0393, -0.0385,  0.0041, -0.0654,  0.0229, -0.0112,  0.0201,
        -0.0191,  0.0215,  0.0067, -0.0664,  0.0883, -0.0541, -0.0795, -0.0317,
        -0.0823,  0.0878, -0.0100,  0.0488, -0.0094,  0.0046,  0.0092,  0.0663],
       requires_grad=True)
----------------------开始执行这行代码----------------------
----------------------结束执行这行代码----------------------
self.encoder[-1].bias.shape:  torch.Size([128])
----------------------我的分割线2----------------------
self.encoder[-1].bias:  Parameter containing:
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)
----------------------结束监视代码----------------------
```
可以看到，**`torch.nn.init.constant_(tensor, val)`函数的作用就是：将`tensor`张量中的所有值都设为`val`，不改变`tensor`张量的形状。** 至于为什么需要把关键点编码器的最后一层的偏置设为0，这个我目前还不是很清楚。之后再研究。
由于现在还不涉及到使用SuperGlue模型进行推理的过程，因此，我暂时先略过模型的`def forward()`函数。之后等用到模型推理的时候，再来分析模型的`def forward()`函数。

