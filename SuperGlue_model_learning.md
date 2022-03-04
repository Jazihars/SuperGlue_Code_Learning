# SuperGlue模型部分的代码学习
在这个笔记中，我将逐行阅读[SuperGlue模型部分的代码](https://github.com/HeatherJiaZG/SuperGlue-pytorch/blob/master/models/superglue.py)，弄懂SuperGlue模型究竟是怎么实现的。

## 从训练脚本中进入模型部分的代码
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

## SuperGlue模型部分的代码逐行分析
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