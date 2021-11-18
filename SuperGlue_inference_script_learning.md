# SuperGlue官方推理代码学习笔记

在这个笔记中，我将对SuperGlue官方推理代码`match_pairs.py`文件进行梳理。弄清楚这份脚本代码的结构和功能。

-----------------------------

我在跑这份`match_pairs.py`脚本文件时，运行的命令是：
``` bash
python match_pairs.py --resize 1600 --superglue outdoor --max_keypoints 100 --nms_radius 3  --resize_float --input_dir assets/mytestphoto --input_pairs /home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestpair.txt --output_dir dump_match_pairs_outdoor --viz
```
在这个命令中，`max_keypoints`参数被设置成了`100`，目的是使可视化出来的效果能够看得比较清楚，不至于有太多的特征点对。

我们首先来看主函数入口处的代码：
``` python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image pair matching and pose evaluation with SuperGlue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_pairs",
        type=str,
        default="assets/scannet_sample_pairs_with_gt.txt",
        help="Path to the list of image pairs",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="assets/scannet_sample_images/",
        help="Path to the directory that contains the images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dump_match_pairs/",
        help="Path to the directory in which the .npz results and optionally,"
        "the visualization images are written",
    )

    parser.add_argument(
        "--max_length", type=int, default=-1, help="Maximum number of pairs to evaluate"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=[640, 480],
        help="Resize the input image before running inference. If two numbers, "
        "resize to the exact dimensions, if one number, resize the max "
        "dimension, if -1, do not resize",
    )
    parser.add_argument(
        "--resize_float",
        action="store_true",
        help="Resize the image after casting uint8 to float",
    )

    parser.add_argument(
        "--superglue",
        choices={"indoor", "outdoor"},
        default="indoor",
        help="SuperGlue weights",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=1024,
        help="Maximum number of keypoints detected by Superpoint"
        " ('-1' keeps all keypoints)",
    )
    parser.add_argument(
        "--keypoint_threshold",
        type=float,
        default=0.005,
        help="SuperPoint keypoint detector confidence threshold",
    )
    parser.add_argument(
        "--nms_radius",
        type=int,
        default=4,
        help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations performed by SuperGlue",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
    )

    parser.add_argument(
        "--viz", action="store_true", help="Visualize the matches and dump the plots"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Perform the evaluation" " (requires ground truth pose and intrinsics)",
    )
    parser.add_argument(
        "--fast_viz",
        action="store_true",
        help="Use faster image visualization with OpenCV instead of Matplotlib",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Skip the pair if output .npz files are already found",
    )
    parser.add_argument(
        "--show_keypoints",
        action="store_true",
        help="Plot the keypoints in addition to the matches",
    )
    parser.add_argument(
        "--viz_extension",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Visualization file extension. Use pdf for highest-quality.",
    )
    parser.add_argument(
        "--opencv_display",
        action="store_true",
        help="Visualize via OpenCV before saving output images",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle ordering of pairs before processing",
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="Force pytorch to run in CPU mode."
    )

    opt = parser.parse_args()
    # print("-------------------开始监视代码----------------------")
    # print("-------------------结束监视代码----------------------")
    print(opt)
```
这些代码的作用和功能是，初始化接下来要用到的各种参数。我们来试运行一下下述代码：
``` python
opt = parser.parse_args()
print("-------------------开始监视代码----------------------")
print(type(opt))
print("-------------------我的分割线1----------------------")
print(opt)
print("-------------------我的分割线2----------------------")
print("opt.max_keypoints：", opt.max_keypoints)
print("-------------------我的分割线3----------------------")
print("opt.viz：", opt.viz)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
<class 'argparse.Namespace'>
-------------------我的分割线1----------------------
Namespace(cache=False, eval=False, fast_viz=False, force_cpu=False, input_dir='assets/mytestphoto', input_pairs='/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestpair.txt', keypoint_threshold=0.005, match_threshold=0.2, max_keypoints=100, max_length=-1, nms_radius=3, opencv_display=False, output_dir='dump_match_pairs_outdoor', resize=[1600], resize_float=True, show_keypoints=False, shuffle=False, sinkhorn_iterations=20, superglue='outdoor', viz=True, viz_extension='png')
-------------------我的分割线2----------------------
opt.max_keypoints： 100
-------------------我的分割线3----------------------
opt.viz： True
-------------------结束监视代码----------------------
```
由此知，初始化的参数被保存在`opt`变量里，调用参数的时候，使用点语法`.`即可。

接下来的四行代码是关于参数的断言：
``` python
assert not (
    opt.opencv_display and not opt.viz
), "Must use --viz with --opencv_display"
assert not (
    opt.opencv_display and not opt.fast_viz
), "Cannot use --opencv_display without --fast_viz"
assert not (opt.fast_viz and not opt.viz), "Must use --viz with --fast_viz"
assert not (
    opt.fast_viz and opt.viz_extension == "pdf"
), "Cannot use pdf extension with --fast_viz"
```
这些断言对参数的初始化增加了一些限制，相当于给参数的初始化增加了四条限制，使得参数并不能被随意地初始化。

接下来的代码是一组`if...else`语句：
``` python
if len(opt.resize) == 2 and opt.resize[1] == -1:
    opt.resize = opt.resize[0:1]
if len(opt.resize) == 2:
    print("Will resize to {}x{} (WxH)".format(opt.resize[0], opt.resize[1]))
elif len(opt.resize) == 1 and opt.resize[0] > 0:
    print("Will resize max dimension to {}".format(opt.resize[0]))
elif len(opt.resize) == 1:
    print("Will not resize images")
else:
    raise ValueError("Cannot specify more than two integers for --resize")
```
这段代码主要是在根据`opt.resize`的不同状态来做不同的处理。我们先来试运行下述代码：
``` python
print("-------------------开始监视代码----------------------")
print(type(opt.resize))
print("-------------------我的分割线1----------------------")
print(opt.resize)
print("-------------------结束监视代码----------------------")
exit()
if len(opt.resize) == 2 and opt.resize[1] == -1:
    opt.resize = opt.resize[0:1]
if len(opt.resize) == 2:
    print("Will resize to {}x{} (WxH)".format(opt.resize[0], opt.resize[1]))
elif len(opt.resize) == 1 and opt.resize[0] > 0:
    print("Will resize max dimension to {}".format(opt.resize[0]))
elif len(opt.resize) == 1:
    print("Will not resize images")
else:
    raise ValueError("Cannot specify more than two integers for --resize")
```
结果为：
```
-------------------开始监视代码----------------------
<class 'list'>
-------------------我的分割线1----------------------
[1600]
-------------------结束监视代码----------------------
```
由此知，`opt.resize`是一个长度为`1`的列表，且包含的唯一的一个值是`1600`。因此，上述`if...else`语句会执行下面的这个分句：
``` python
elif len(opt.resize) == 1 and opt.resize[0] > 0:
    print("Will resize max dimension to {}".format(opt.resize[0]))
```
经测试，的确如此。

接下来是这样的两行代码：
``` python
with open(opt.input_pairs, "r") as f:
    pairs = [l.split() for l in f.readlines()]
```
这两行代码是在打开一个文件，从文件中提取某些信息。我们先来测试一下这两行代码要打开的是什么文件：
``` python
print("-------------------开始监视代码----------------------")
print(type(opt.input_pairs))
print("-------------------我的分割线1----------------------")
print(opt.input_pairs)
print("-------------------结束监视代码----------------------")
exit()
with open(opt.input_pairs, "r") as f:
    pairs = [l.split() for l in f.readlines()]
```
结果为：
```
-------------------开始监视代码----------------------
<class 'str'>
-------------------我的分割线1----------------------
/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestpair.txt
-------------------结束监视代码----------------------
```
可以看到，这两行代码要打开的是`/SuperGluePretrainedNetwork/assets/mytestpair.txt`文件。我的`/SuperGluePretrainedNetwork/assets/mytestpair.txt`文件中只有两行内容，这两行内容如下：
```
/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_1.png /home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_2.png
/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo2_1.png /home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo2_2.png
```
再来测试一下下面的代码：
``` python
with open(opt.input_pairs, "r") as f:
    pairs = [l.split() for l in f.readlines()]
print("-------------------开始监视代码----------------------")
print(type(pairs))
print("-------------------我的分割线1----------------------")
print(pairs)
print("-------------------结束监视代码----------------------")
exit()
```
结果如下：
```
-------------------开始监视代码----------------------
<class 'list'>
-------------------我的分割线1----------------------
[['/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_1.png', '/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_2.png'], ['/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo2_1.png', '/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo2_2.png']]
-------------------结束监视代码----------------------
```
至此，已经完全弄清楚了这两行读取文件的代码的功能。这两行读取文件的代码，最终目的就是，把之后要匹配的图片的绝对路径放到一个列表里。每个匹配图片对的绝对路径组成的列表再被放到一个大的列表里。

接下来是这样的两行代码：
``` python
if opt.max_length > -1:
    pairs = pairs[0 : np.min([len(pairs), opt.max_length])]
```
试运行如下的代码：
``` python
print("-------------------开始监视代码----------------------")
print(opt.max_length)
print("-------------------结束监视代码----------------------")
exit()
if opt.max_length > -1:
    pairs = pairs[0 : np.min([len(pairs), opt.max_length])]
```
结果为：
```
-------------------开始监视代码----------------------
-1
-------------------结束监视代码----------------------
```
由此可知，由`if`语句包裹的这行代码不会被执行。

测试如下的代码：
``` python
print("-------------------开始监视代码----------------------")
print(opt.shuffle)
print("-------------------结束监视代码----------------------")
exit()
if opt.shuffle:
    random.Random(0).shuffle(pairs)
```
结果为：
```
-------------------开始监视代码----------------------
False
-------------------结束监视代码----------------------
```
由此知，程序在运行时，不会做这一步随机变换。

再测试如下的代码：
``` python
print("-------------------开始监视代码----------------------")
print(opt.eval)
print("-------------------结束监视代码----------------------")
exit()
if opt.eval:
    if not all([len(p) == 38 for p in pairs]):
        raise ValueError(
            "All pairs should have ground truth info for evaluation."
            'File "{}" needs 38 valid entries per row'.format(opt.input_pairs)
        )
```
结果为：
```
-------------------开始监视代码----------------------
False
-------------------结束监视代码----------------------
```
由此知，这一段由`if opt.eval:`包裹的代码不会被执行。

直接运行主程序的下面两行代码：
``` python
device = "cuda" if torch.cuda.is_available() and not opt.force_cpu else "cpu"
print('Running inference on device "{}"'.format(device))
exit()
```
结果为：
```
Running inference on device "cuda"
```
这说明，我用的设备是GPU而不是CPU。

接下来，试运行下述代码：
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
print("-------------------开始监视代码----------------------")
print(type(config))
print("-------------------我的分割线1----------------------")
print(config)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
<class 'dict'>
-------------------我的分割线1----------------------
{'superpoint': {'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': 100}, 'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}}
-------------------结束监视代码----------------------
```
由此知，`config`对象是一个用字典数据类型来存储的参数集合。之后调用`config`中存储的参数时，使用字典的中括号`[key]`语法即可。

接下来是这样的一行代码：
``` python
matching = Matching(config).eval().to(device)
```
这行代码是整个推理脚本的第一个关键核心代码。这行代码执行了整个推理脚本所必须的一个关键功能：使用已经训练好的模型的权重，来初始化我们接下来的推理所要用到的网络。我们先来看看它的两个输入`config`和`device`。按照我的预期，初始化网络所要用到的权重文件的路径就应该保存在这个`config`变量里。先来看看我的猜想对不对。试运行下述代码：
``` python
print("-------------------开始监视代码----------------------")
print(type(config))
print("-------------------我的分割线1----------------------")
print(config)
print("-------------------我的分割线2----------------------")
print(device == "cuda")
print("-------------------结束监视代码----------------------")
exit()
matching = Matching(config).eval().to(device)
```
结果为：
```
-------------------开始监视代码----------------------
<class 'dict'>
-------------------我的分割线1----------------------
{'superpoint': {'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': 100}, 'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}}
-------------------我的分割线2----------------------
True
-------------------结束监视代码----------------------
```
由此知，输入的`config`是一个字典。这个`config`字典里存储了初始化`superpoint`网络和`superglue`网络的参数。`device`是字符串`"cuda"`。并且还可以看到，我的猜想不正确。初始化网络所要用到的权重文件的路径，并没有被保存在`config`变量里。接下来，我必须得弄清楚，到底是在哪里使用了论文作者已经训练好的权重文件，以及权重文件的路径究竟是被存储在哪里的。进入`Matching`类的定义，在`/SuperGluePretrainedNetwork/models/superglue.py`文件的`class SuperGlue(nn.Module): def __init__(self, config):`函数里，可以看到下面的代码：
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
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(str(path)))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))
```
特别来看一下`def __init__(self, config):`函数里的这一段：
``` python
assert self.config['weights'] in ['indoor', 'outdoor']
path = Path(__file__).parent
path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
self.load_state_dict(torch.load(str(path)))
print('Loaded SuperGlue model (\"{}\" weights)'.format(
    self.config['weights']))
```
至此，终于明白了。在`/SuperGluePretrainedNetwork/models/superglue.py`文件的`class SuperGlue(nn.Module): def __init__(self, config):`函数里，加载了论文作者已经训练好的模型权重。并且，权重文件的路径已经被写死在`class SuperGlue(nn.Module): def __init__(self, config):`函数里了。在`/SuperGluePretrainedNetwork/models/superglue.py`文件里，测试如下的代码：
``` python
assert self.config["weights"] in ["indoor", "outdoor"]
path = Path(__file__).parent
path = path / "weights/superglue_{}.pth".format(self.config["weights"])
print("-------------------开始监视代码----------------------")
print("type(path)：", type(path))
print("-------------------我的分割线1----------------------")
print("path：", path)
print("-------------------结束监视代码----------------------")
exit()
self.load_state_dict(torch.load(str(path)))
print('Loaded SuperGlue model ("{}" weights)'.format(self.config["weights"]))
```
结果为：
```
-------------------开始监视代码----------------------
type(path)： <class 'pathlib.PosixPath'>
-------------------我的分割线1----------------------
path： /home/users/zitong.yin/SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth
-------------------结束监视代码----------------------
```
此时，我就完全弄清楚了初始化网络所要用到的论文作者提供的权重文件的路径。权重文件的路径是被写死在网络初始化的代码里的。
初始化网络的代码中，还用到了一些PyTorch函数的用法，这些以后再抽时间来学习吧。

接下来，我们再来看看这行初始化网络的代码`matching = Matching(config).eval().to(device)`的输出：`matching`变量。我要弄清楚初始化网络输出的这个`matching`变量究竟是什么。弄懂这个变量，才能知道之后要怎样来使用初始化好的网络。试运行下述代码：
``` python
matching = Matching(config).eval().to(device)
print("-------------------开始监视代码----------------------")
print(type(matching))
print("-------------------我的分割线1----------------------")
print(matching)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
<class 'models.matching.Matching'>
-------------------我的分割线1----------------------
Matching(
  (superpoint): SuperPoint(
    (relu): ReLU(inplace=True)
    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv1a): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv1b): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2a): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2b): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3a): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3b): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4a): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4b): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (convPa): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (convPb): Conv2d(256, 65, kernel_size=(1, 1), stride=(1, 1))
    (convDa): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (convDb): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (superglue): SuperGlue(
    (kenc): KeypointEncoder(
      (encoder): Sequential(
        (0): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
        (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv1d(32, 64, kernel_size=(1,), stride=(1,))
        (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
        (7): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
        (9): Conv1d(128, 256, kernel_size=(1,), stride=(1,))
        (10): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (11): ReLU()
        (12): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
      )
    )
    (gnn): AttentionalGNN(
      (layers): ModuleList(
        (0): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (1): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (2): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (3): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (4): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (5): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (6): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (7): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (8): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (9): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (10): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (11): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (12): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (13): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (14): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (15): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (16): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (17): AttentionalPropagation(
          (attn): MultiHeadedAttention(
            (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (proj): ModuleList(
              (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
              (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            )
          )
          (mlp): Sequential(
            (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
          )
        )
      )
    )
    (final_proj): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
  )
)
-------------------结束监视代码----------------------
```
至此，`SuperGlue`模型终于展现出了它的全貌了。我使用的推理模型`SuperGlue`是存储在变量`matching`里的。当把变量`matching`print出来之后，可以看到，我使用的推理模型`SuperGlue`到底是由哪些具体的网络层组成的。`matching`变量里存储了我使用的推断网络的网络结构细节。至于之后要怎样具体地使用网络，那就需要继续深入分析接下来的代码才能知晓了。

接下来是这样的一段代码：
``` python
# Create the output directories if they do not exist already.
input_dir = Path(opt.input_dir)
print('Looking for data in directory "{}"'.format(input_dir))
output_dir = Path(opt.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory "{}"'.format(output_dir))
if opt.eval:
    print("Will write evaluation results", 'to directory "{}"'.format(output_dir))
if opt.viz:
    print("Will write visualization images to", 'directory "{}"'.format(output_dir))
```
这段代码的功能是：创建输入数据的路径对象和输出数据的路径对象。我们先来测试一下下述代码：
``` python
# Create the output directories if they do not exist already.
print("-------------------开始监视代码----------------------")
print("type(opt.input_dir)：", type(opt.input_dir))
print("-------------------我的分割线1----------------------")
print("opt.input_dir：", opt.input_dir)
print("-------------------我的分割线2----------------------")
print("type(opt.output_dir)：", type(opt.output_dir))
print("-------------------我的分割线3----------------------")
print("opt.output_dir：", opt.output_dir)
print("-------------------结束监视代码----------------------")
exit()
input_dir = Path(opt.input_dir)
print('Looking for data in directory "{}"'.format(input_dir))
output_dir = Path(opt.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory "{}"'.format(output_dir))
if opt.eval:
    print("Will write evaluation results", 'to directory "{}"'.format(output_dir))
if opt.viz:
    print("Will write visualization images to", 'directory "{}"'.format(output_dir))
```
结果为：
```
-------------------开始监视代码----------------------
type(opt.input_dir)： <class 'str'>
-------------------我的分割线1----------------------
opt.input_dir： assets/mytestphoto
-------------------我的分割线2----------------------
type(opt.output_dir)： <class 'str'>
-------------------我的分割线3----------------------
opt.output_dir： dump_match_pairs_outdoor
-------------------结束监视代码----------------------
```
要注意一点：我的`opt.input_dir`和`opt.output_dir`参数给出的都是相对路径。当前路径是`/SuperGluePretrainedNetwork`。
再来试运行下述代码：
``` python
# Create the output directories if they do not exist already.
input_dir = Path(opt.input_dir)
print("-------------------开始监视代码----------------------")
print(type(input_dir))
print("-------------------我的分割线1----------------------")
print(input_dir)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
<class 'pathlib.PosixPath'>
-------------------我的分割线1----------------------
assets/mytestphoto
-------------------结束监视代码----------------------
```
`Path`对象是由这份代码开头的`from pathlib import Path`语句引入的。关于Python标准库`pathlib`的用法，参见[Python官方文档](https://docs.python.org/3.7/library/pathlib.html)。也就是说，`input_dir = Path(opt.input_dir)`这行代码的作用就是，把一个字符串路径转换成了一个Python标准库`<class 'pathlib.PosixPath'>`类型的路径。这个步骤的具体作用，下面会揭晓。

依次运行下述的两段代码：
``` python
# Create the output directories if they do not exist already.
input_dir = Path(opt.input_dir)
print('Looking for data in directory "{}"'.format(input_dir))
output_dir = Path(opt.output_dir)
exit()
output_dir.mkdir(exist_ok=True, parents=True)
# exit()
```
``` python
# Create the output directories if they do not exist already.
input_dir = Path(opt.input_dir)
print('Looking for data in directory "{}"'.format(input_dir))
output_dir = Path(opt.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
exit()
```
可以看出，`output_dir.mkdir(exist_ok=True, parents=True)`这行代码在当前路径`/SuperGluePretrainedNetwork`之下创建了一个新的子路径`/SuperGluePretrainedNetwork/dump_match_pairs_outdoor`。这个路径目前还是刚刚被创建，是一个空路径。

最后，再来试运行下述代码：
``` python
# Create the output directories if they do not exist already.
print("-------------------开始监视代码----------------------")
input_dir = Path(opt.input_dir)
print('Looking for data in directory "{}"'.format(input_dir))
output_dir = Path(opt.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory "{}"'.format(output_dir))
if opt.eval:
    print("Will write evaluation results", 'to directory "{}"'.format(output_dir))
if opt.viz:
    print("Will write visualization images to", 'directory "{}"'.format(output_dir))
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
Looking for data in directory "assets/mytestphoto"
Will write matches to directory "dump_match_pairs_outdoor"
Will write visualization images to directory "dump_match_pairs_outdoor"
-------------------结束监视代码----------------------
```
由此知，我本次对`match_pairs.py`脚本的运行，不会执行`eval`（评估）的过程，只会执行`visualization`（可视化）的过程，并且会把匹配之后的结果和可视化之后的结果写到`/SuperGluePretrainedNetwork/dump_match_pairs_outdoor`路径里。

接下来是这样的一行代码：
``` python
timer = AverageTimer(newline=True)
```
这个`timer`和监控代码的运行时间有关。我们来试运行下述代码：
``` python
timer = AverageTimer(newline=True)
print("-------------------开始监视代码----------------------")
print(type(timer))
print("-------------------我的分割线1----------------------")
print(timer)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
<class 'models.utils.AverageTimer'>
-------------------我的分割线1----------------------
<models.utils.AverageTimer object at 0x7f6b15b0bf90>
-------------------结束监视代码----------------------
```
这个`timer`变量是一个`AverageTimer`类的实例。我们进入到`/SuperGluePretrainedNetwork/models/utils.py`文件中看一下`AverageTimer`类的定义：
``` python
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()
```
这个类的定义，目前我暂时不需要详细地研究。之后如果用到了`timer`变量的相关用法，不会了，再来学习这个`AverageTimer`类的用法。

接下来我们就进入了整个推理脚本最核心的部分，即下面的这个`for`循环：
``` python
for i, pair in enumerate(pairs):
    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    matches_path = output_dir / "{}_{}_matches.npz".format(stem0, stem1)
    eval_path = output_dir / "{}_{}_evaluation.npz".format(stem0, stem1)
    viz_path = output_dir / "{}_{}_matches.{}".format(
        stem0, stem1, opt.viz_extension
    )
    viz_eval_path = output_dir / "{}_{}_evaluation.{}".format(
        stem0, stem1, opt.viz_extension
    )

    # Handle --cache logic.
    do_match = True
    do_eval = opt.eval
    do_viz = opt.viz
    do_viz_eval = opt.eval and opt.viz
    if opt.cache:
        if matches_path.exists():
            try:
                results = np.load(matches_path)
            except:
                raise IOError("Cannot load matches .npz file: %s" % matches_path)

            kpts0, kpts1 = results["keypoints0"], results["keypoints1"]
            matches, conf = results["matches"], results["match_confidence"]
            do_match = False
        if opt.eval and eval_path.exists():
            try:
                results = np.load(eval_path)
            except:
                raise IOError("Cannot load eval .npz file: %s" % eval_path)
            err_R, err_t = results["error_R"], results["error_t"]
            precision = results["precision"]
            matching_score = results["matching_score"]
            num_correct = results["num_correct"]
            epi_errs = results["epipolar_errors"]
            do_eval = False
        if opt.viz and viz_path.exists():
            do_viz = False
        if opt.viz and opt.eval and viz_eval_path.exists():
            do_viz_eval = False
        timer.update("load_cache")

    if not (do_match or do_eval or do_viz or do_viz_eval):
        timer.print("Finished pair {:5} of {:5}".format(i, len(pairs)))
        continue

    # If a rotation integer is provided (e.g. from EXIF data), use it:
    if len(pair) >= 5:
        rot0, rot1 = int(pair[2]), int(pair[3])
    else:
        rot0, rot1 = 0, 0

    # Load the image pair.
    image0, inp0, scales0 = read_image(
        input_dir / name0, device, opt.resize, rot0, opt.resize_float
    )
    image1, inp1, scales1 = read_image(
        input_dir / name1, device, opt.resize, rot1, opt.resize_float
    )
    if image0 is None or image1 is None:
        print(
            "Problem reading image pair: {} {}".format(
                input_dir / name0, input_dir / name1
            )
        )
        exit(1)
    timer.update("load_image")

    if do_match:
        # Perform the matching.
        pred = matching({"image0": inp0, "image1": inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches, conf = pred["matches0"], pred["matching_scores0"]
        timer.update("matcher")

        # Write the matches to disk.
        out_matches = {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "matches": matches,
            "match_confidence": conf,
        }
        np.savez(str(matches_path), **out_matches)

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    if do_eval:
        # Estimate the pose and compute the pose error.
        assert len(pair) == 38, "Pair does not have ground truth info"
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(K0, scales0)
        K1 = scale_intrinsics(K1, scales1)

        # Update the intrinsics + extrinsics if EXIF rotation was found.
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(K0, image0.shape, rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(K1, image1.shape, rot1)
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0

        epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
        correct = epi_errs < 5e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

        thresh = 1.0  # In pixels relative to resized image size.
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        # Write the evaluation results to disk.
        out_eval = {
            "error_t": err_t,
            "error_R": err_R,
            "precision": precision,
            "matching_score": matching_score,
            "num_correct": num_correct,
            "epipolar_errors": epi_errs,
        }
        np.savez(str(eval_path), **out_eval)
        timer.update("eval")

    # 这个if语句所包裹的代码实现了画出可视化的匹配点
    if do_viz:
        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
            "Matches: {}".format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append("Rotation: {}:{}".format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config["keypoint_threshold"]
        m_thresh = matching.superglue.config["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
            "Image Pair: {}:{}".format(stem0, stem1),
        ]

        make_matching_plot(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            viz_path,
            opt.show_keypoints,
            opt.fast_viz,
            opt.opencv_display,
            "Matches",
            small_text,
        )

        timer.update("viz_match")

    if do_viz_eval:
        # Visualize the evaluation results for the image pair.
        color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
        color = error_colormap(1 - color)
        deg, delta = " deg", "Delta "
        if not opt.fast_viz:
            deg, delta = "°", "$\\Delta$"
        e_t = "FAIL" if np.isinf(err_t) else "{:.1f}{}".format(err_t, deg)
        e_R = "FAIL" if np.isinf(err_R) else "{:.1f}{}".format(err_R, deg)
        text = [
            "SuperGlue",
            "{}R: {}".format(delta, e_R),
            "{}t: {}".format(delta, e_t),
            "inliers: {}/{}".format(num_correct, (matches > -1).sum()),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append("Rotation: {}:{}".format(rot0, rot1))

        # Display extra parameter info (only works with --fast_viz).
        k_thresh = matching.superpoint.config["keypoint_threshold"]
        m_thresh = matching.superglue.config["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
            "Image Pair: {}:{}".format(stem0, stem1),
        ]

        make_matching_plot(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            viz_eval_path,
            opt.show_keypoints,
            opt.fast_viz,
            opt.opencv_display,
            "Relative Pose",
            small_text,
        )

        timer.update("viz_eval")

    timer.print("Finished pair {:5} of {:5}".format(i, len(pairs)))
```
这个`for`循环是整个`/SuperGluePretrainedNetwork/match_pairs.py`脚本的核心部分。我们必须非常细致地予以分析。
首先，我们先来看看`for`循环会进行多少轮。测试如下的代码：
``` python
print("-------------------开始监视代码----------------------")
print(pairs)
print("-------------------我的分割线1----------------------")
for i, pair in enumerate(pairs):
    print(pair)
    print(f"结束了第{i+1}轮循环")
print("-------------------结束监视代码----------------------")
exit()
for i, pair in enumerate(pairs):
    name0, name1 = pair[:2]
    stem0, stem1 = Path(name0).stem, Path(name1).stem
```
结果为：
```
-------------------开始监视代码----------------------
[['/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_1.png', '/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_2.png'], ['/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo2_1.png', '/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo2_2.png']]
-------------------我的分割线1----------------------
['/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_1.png', '/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_2.png']
结束了第1轮循环
['/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo2_1.png', '/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo2_2.png']
结束了第2轮循环
-------------------结束监视代码----------------------
```
由此，我们知道了，因为我只给了两对图片对，所以这个`for`循环会进行两轮。`pair`这个变量会在接下来的`for`循环内部用来表示要匹配的图片对的绝对路径。

接下来试运行下述代码：
``` python
for i, pair in enumerate(pairs):
    name0, name1 = pair[:2]
    print("-------------------开始监视代码----------------------")
    print(name0)
    print("-------------------我的分割线1----------------------")
    print(name1)
    print("-------------------结束监视代码----------------------")
    exit()
```
结果为：
```
-------------------开始监视代码----------------------
/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_1.png
-------------------我的分割线1----------------------
/home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_2.png
-------------------结束监视代码----------------------
```
这样一来就完全清楚了。`name0`和`name1`这两个变量里存储的就是两张匹配图像的绝对路径。

接下来测试一下下述代码（注意：下述代码都是在`for`循环中的）：
``` python
stem0, stem1 = Path(name0).stem, Path(name1).stem
print("-------------------开始监视代码----------------------")
print(type(stem0))
print("-------------------我的分割线1----------------------")
print(stem0)
print("-------------------我的分割线2----------------------")
print(type(stem1))
print("-------------------我的分割线3----------------------")
print(stem1)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
<class 'str'>
-------------------我的分割线1----------------------
photo1_1
-------------------我的分割线2----------------------
<class 'str'>
-------------------我的分割线3----------------------
photo1_2
-------------------结束监视代码----------------------
```
至此，我明白了Python标准库`Path`的`Path(str).stem`函数的用法。在一个空的Python脚本里测试如下的代码（注意：我是在`Python 3.7.12`中测试的）：
``` python
from pathlib import Path

x = "/aaaa/bbbb/cccc/asdfg.sdfs"
print(Path(x).stem == "asdfg")
```
结果为：
```
True
```
这就完全清楚了。`Path(str).stem`的目的是，提取出一个文件路径中的文件名，并且去掉扩展名。

接下来试运行下述代码：
``` python
print("-------------------开始监视代码----------------------")
print(type(output_dir))
print("-------------------我的分割线1----------------------")
print(output_dir)
print("-------------------我的分割线2----------------------")
print(type(opt.viz_extension))
print("-------------------我的分割线3----------------------")
print(opt.viz_extension)
print("-------------------结束监视代码----------------------")
exit()
matches_path = output_dir / "{}_{}_matches.npz".format(stem0, stem1)
eval_path = output_dir / "{}_{}_evaluation.npz".format(stem0, stem1)
viz_path = output_dir / "{}_{}_matches.{}".format(
    stem0, stem1, opt.viz_extension
)
viz_eval_path = output_dir / "{}_{}_evaluation.{}".format(
    stem0, stem1, opt.viz_extension
)
```
结果为：
```
-------------------开始监视代码----------------------
<class 'pathlib.PosixPath'>
-------------------我的分割线1----------------------
dump_match_pairs_outdoor
-------------------我的分割线2----------------------
<class 'str'>
-------------------我的分割线3----------------------
png
-------------------结束监视代码----------------------
```
这样，就清楚了接下来要用到的`output_dir`参数和`opt.viz_extension`参数的类型和值。

接下来试运行下述代码：
``` python
matches_path = output_dir / "{}_{}_matches.npz".format(stem0, stem1)
eval_path = output_dir / "{}_{}_evaluation.npz".format(stem0, stem1)
viz_path = output_dir / "{}_{}_matches.{}".format(
    stem0, stem1, opt.viz_extension
)
viz_eval_path = output_dir / "{}_{}_evaluation.{}".format(
    stem0, stem1, opt.viz_extension
)
print("-------------------开始监视代码----------------------")
print("type(matches_path)：", type(matches_path))
print("-------------------我的分割线1----------------------")
print("matches_path：", matches_path)
print("-------------------我的分割线2----------------------")
print("type(eval_path)：", type(eval_path))
print("-------------------我的分割线3----------------------")
print("eval_path：", eval_path)
print("-------------------我的分割线4----------------------")
print("type(viz_path)：", type(viz_path))
print("-------------------我的分割线5----------------------")
print("viz_path：", viz_path)
print("-------------------我的分割线6----------------------")
print("type(viz_eval_path)：", type(viz_eval_path))
print("-------------------我的分割线7----------------------")
print("viz_eval_path：", viz_eval_path)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
type(matches_path)： <class 'pathlib.PosixPath'>
-------------------我的分割线1----------------------
matches_path： dump_match_pairs_outdoor/photo1_1_photo1_2_matches.npz
-------------------我的分割线2----------------------
type(eval_path)： <class 'pathlib.PosixPath'>
-------------------我的分割线3----------------------
eval_path： dump_match_pairs_outdoor/photo1_1_photo1_2_evaluation.npz
-------------------我的分割线4----------------------
type(viz_path)： <class 'pathlib.PosixPath'>
-------------------我的分割线5----------------------
viz_path： dump_match_pairs_outdoor/photo1_1_photo1_2_matches.png
-------------------我的分割线6----------------------
type(viz_eval_path)： <class 'pathlib.PosixPath'>
-------------------我的分割线7----------------------
viz_eval_path： dump_match_pairs_outdoor/photo1_1_photo1_2_evaluation.png
-------------------结束监视代码----------------------
```
至此，我已经完全搞清楚了利用Python标准库对路径进行`/`运算的用法。在空白脚本中测试下述代码（注意：我是在`Python 3.7.12`中测试的）：
``` python
from pathlib import Path

x = "/aaaa/bbbb/cccc"
y = Path(x)
z = y / "asdfgzxc"
w = z / "qwert.zxc"

print("type(y)：", type(y))
print("type(z)：", type(z))
print("type(w)：", type(w))
print("z：", z)
print("w：", w)
```
结果为：
```
type(y)： <class 'pathlib.PosixPath'>
type(z)： <class 'pathlib.PosixPath'>
type(w)： <class 'pathlib.PosixPath'>
z： /aaaa/bbbb/cccc/asdfgzxc
w： /aaaa/bbbb/cccc/asdfgzxc/qwert.zxc
```
通过这个测试，我明白了：Python标准库中的`Path`模块可以把一个字符串转换成`<class 'pathlib.PosixPath'>`类型的路径。`<class 'pathlib.PosixPath'>`类型的路径可以直接和字符串进行`/`运算，这种方法可以更简便地来构建新的路径。

接下来是一些逻辑值的设定。这些逻辑值控制了接下来的代码中，哪些部分需要运行，哪些部分不需要运行。测试如下的代码：
``` python
# Handle --cache logic.
do_match = True
do_eval = opt.eval
do_viz = opt.viz
do_viz_eval = opt.eval and opt.viz
print("-------------------开始监视代码----------------------")
print("do_eval：", do_eval)
print("-------------------我的分割线1----------------------")
print("do_viz：", do_viz)
print("-------------------我的分割线2----------------------")
print("do_viz_eval：", do_viz_eval)
print("-------------------我的分割线3----------------------")
print("opt.cache：", opt.cache)
print("-------------------结束监视代码----------------------")
exit()
if opt.cache:
    if matches_path.exists():
```
结果为：
```
-------------------开始监视代码----------------------
do_eval： False
-------------------我的分割线1----------------------
do_viz： True
-------------------我的分割线2----------------------
do_viz_eval： False
-------------------我的分割线3----------------------
opt.cache： False
-------------------结束监视代码----------------------
```
之后在判断哪些代码快需要运行时，就需要来参考这里的这些逻辑值。
因为`opt.cache`的值为`False`，所以下面由`if opt.cache:`语句包裹的代码块不需要执行。

接下来进行如下的测试：
``` python
print("-------------------开始监视代码----------------------")
print(not (do_match or do_eval or do_viz or do_viz_eval))
print("-------------------结束监视代码----------------------")
exit()
if not (do_match or do_eval or do_viz or do_viz_eval):
    timer.print("Finished pair {:5} of {:5}".format(i, len(pairs)))
    continue
```
结果为：
```
-------------------开始监视代码----------------------
False
-------------------结束监视代码----------------------
```
由此知，这个`if`语句也不需要执行。

接下来试运行下述代码：
``` python
# If a rotation integer is provided (e.g. from EXIF data), use it:
print("-------------------开始监视代码----------------------")
print("len(pair)：", len(pair))
print("-------------------结束监视代码----------------------")
exit()
if len(pair) >= 5:
    rot0, rot1 = int(pair[2]), int(pair[3])
else:
    rot0, rot1 = 0, 0
```
结果为：
```
-------------------开始监视代码----------------------
len(pair)： 2
-------------------结束监视代码----------------------
```
由此知，`rot0`和`rot1`都被设为`0`

接下来就进入了读取图像的代码了。试运行下述代码：
``` python
# Load the image pair.
print("-------------------开始监视代码----------------------")
print("input_dir / name0：", input_dir / name0)
print("-------------------我的分割线1----------------------")
print("input_dir / name1：", input_dir / name1)
print("-------------------我的分割线2----------------------")
print(device == "cuda")
print("-------------------我的分割线3----------------------")
print("opt.resize：", opt.resize)
print("-------------------我的分割线4----------------------")
print("opt.resize_float：", opt.resize_float)
print("-------------------结束监视代码----------------------")
exit()
image0, inp0, scales0 = read_image(
    input_dir / name0, device, opt.resize, rot0, opt.resize_float
)
image1, inp1, scales1 = read_image(
    input_dir / name1, device, opt.resize, rot1, opt.resize_float
)
```
结果为：
```
-------------------开始监视代码----------------------
input_dir / name0： /home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_1.png
-------------------我的分割线1----------------------
input_dir / name1： /home/users/zitong.yin/SuperGluePretrainedNetwork/assets/mytestphoto/photo1_2.png
-------------------我的分割线2----------------------
True
-------------------我的分割线3----------------------
opt.resize： [1600]
-------------------我的分割线4----------------------
opt.resize_float： True
-------------------结束监视代码----------------------
```
这些就是`read_image(...)`函数需要用到的参数。上面已经说了，`rot0`和`rot1`都是`0`。

下面我们以第一张图像为例，来看一下`read_image(...)`函数究竟输出了什么。试运行下述代码：
``` python
# Load the image pair.
image0, inp0, scales0 = read_image(
    input_dir / name0, device, opt.resize, rot0, opt.resize_float
)
print("-------------------开始监视代码----------------------")
print("type(image0)：", type(image0))
print("-------------------我的分割线1----------------------")
print("image0.shape：", image0.shape)
print("-------------------我的分割线2----------------------")
print("image0：", image0)
print("-------------------我的分割线3----------------------")
print("type(inp0)：", type(inp0))
print("-------------------我的分割线4----------------------")
print("inp0.shape：", inp0.shape)
print("-------------------我的分割线5----------------------")
print("inp0：", inp0)
print("-------------------我的分割线6----------------------")
print("type(scales0)：", type(scales0))
print("-------------------我的分割线7----------------------")
print("scales0：", scales0)
print("-------------------结束监视代码----------------------")
exit()
image1, inp1, scales1 = read_image(
    input_dir / name1, device, opt.resize, rot1, opt.resize_float
)
```
结果为（这些是第一张图像的数据结构）：
```
-------------------开始监视代码----------------------
type(image0)： <class 'numpy.ndarray'>
-------------------我的分割线1----------------------
image0.shape： (900, 1600)
-------------------我的分割线2----------------------
image0： [[ 22.29      20.41      22.85     ...  41.        41.        41.      ]
 [ 22.85      10.279999  17.85     ...  42.        41.100002  42.      ]
 [ 22.65      22.400002  21.       ...  42.25      42.        42.      ]
 ...
 [ 88.        88.        88.       ... 109.       108.799805 108.69995 ]
 [ 88.        88.        88.       ... 109.       108.799805 108.69995 ]
 [ 88.        88.        88.       ... 109.       108.799805 108.69995 ]]
-------------------我的分割线3----------------------
type(inp0)： <class 'torch.Tensor'>
-------------------我的分割线4----------------------
inp0.shape： torch.Size([1, 1, 900, 1600])
-------------------我的分割线5----------------------
inp0： tensor([[[[0.0874, 0.0800, 0.0896,  ..., 0.1608, 0.1608, 0.1608],
          [0.0896, 0.0403, 0.0700,  ..., 0.1647, 0.1612, 0.1647],
          [0.0888, 0.0878, 0.0824,  ..., 0.1657, 0.1647, 0.1647],
          ...,
          [0.3451, 0.3451, 0.3451,  ..., 0.4275, 0.4267, 0.4263],
          [0.3451, 0.3451, 0.3451,  ..., 0.4275, 0.4267, 0.4263],
          [0.3451, 0.3451, 0.3451,  ..., 0.4275, 0.4267, 0.4263]]]],
       device='cuda:0')
-------------------我的分割线6----------------------
type(scales0)： <class 'tuple'>
-------------------我的分割线7----------------------
scales0： (2.4, 2.4)
-------------------结束监视代码----------------------
```
再来看看第二张图像的`read_image(...)`输出数据结构。运行下述代码：
``` python
image1, inp1, scales1 = read_image(
    input_dir / name1, device, opt.resize, rot1, opt.resize_float
)
print("-------------------开始监视代码----------------------")
print("type(image1)：", type(image1))
print("-------------------我的分割线1----------------------")
print("image1.shape：", image1.shape)
print("-------------------我的分割线2----------------------")
print("image1：", image1)
print("-------------------我的分割线3----------------------")
print("type(inp1)：", type(inp1))
print("-------------------我的分割线4----------------------")
print("inp1.shape：", inp1.shape)
print("-------------------我的分割线5----------------------")
print("inp1：", inp1)
print("-------------------我的分割线6----------------------")
print("type(scales1)：", type(scales1))
print("-------------------我的分割线7----------------------")
print("scales1：", scales1)
print("-------------------结束监视代码----------------------")
exit()
```
结果为（这些是第二张图像的数据结构）：
```
-------------------开始监视代码----------------------
type(image1)： <class 'numpy.ndarray'>
-------------------我的分割线1----------------------
image1.shape： (900, 1600)
-------------------我的分割线2----------------------
image1： [[ 22.02      15.7       18.7      ...  44.65      45.        43.489967]
 [ 21.05       6.569998  13.049999 ...  44.        43.899902  44.100002]
 [ 13.8       19.8       18.75     ...  44.        44.        44.849976]
 ...
 [ 94.        94.        94.       ... 113.5      114.799805 113.      ]
 [ 94.        94.        94.       ... 113.5      114.799805 113.      ]
 [ 94.        94.        94.       ... 113.5      114.799805 113.      ]]
-------------------我的分割线3----------------------
type(inp1)： <class 'torch.Tensor'>
-------------------我的分割线4----------------------
inp1.shape： torch.Size([1, 1, 900, 1600])
-------------------我的分割线5----------------------
inp1： tensor([[[[0.0864, 0.0616, 0.0733,  ..., 0.1751, 0.1765, 0.1705],
          [0.0825, 0.0258, 0.0512,  ..., 0.1725, 0.1722, 0.1729],
          [0.0541, 0.0776, 0.0735,  ..., 0.1725, 0.1725, 0.1759],
          ...,
          [0.3686, 0.3686, 0.3686,  ..., 0.4451, 0.4502, 0.4431],
          [0.3686, 0.3686, 0.3686,  ..., 0.4451, 0.4502, 0.4431],
          [0.3686, 0.3686, 0.3686,  ..., 0.4451, 0.4502, 0.4431]]]],
       device='cuda:0')
-------------------我的分割线6----------------------
type(scales1)： <class 'tuple'>
-------------------我的分割线7----------------------
scales1： (2.4, 2.4)
-------------------结束监视代码----------------------
```

由此就明白了：`read_image(path, device, resize, rotation, resize_float)`这个函数接收一张图片路径的输入，输出这张图片的`numpy`和`torch.tensor`数据格式。之后，要输入到SuperGlue网络中的就是两张匹配图像对的`numpy`和`torch.tensor`数据格式。

关于图片对的加载，最后再来试运行一下下述代码：
``` python
# Load the image pair.
image0, inp0, scales0 = read_image(
    input_dir / name0, device, opt.resize, rot0, opt.resize_float
)
image1, inp1, scales1 = read_image(
    input_dir / name1, device, opt.resize, rot1, opt.resize_float
)
if image0 is None or image1 is None:
    print(
        "Problem reading image pair: {} {}".format(
            input_dir / name0, input_dir / name1
        )
    )
    exit(1)
timer.update("load_image")
print("-------------------开始监视代码----------------------")
print(type(timer))
print("-------------------我的分割线1----------------------")
print(timer)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
<class 'models.utils.AverageTimer'>
-------------------我的分割线1----------------------
<models.utils.AverageTimer object at 0x7f61acb12bd0>
-------------------结束监视代码----------------------
```
观察终端的输出可知，我的终端里没有print出`"Problem reading image pair...`的内容，这就说明，读取图片对的过程是正常的。

接下来遇到了`if do_match:`语句包裹的代码。我们先来试运行下述代码：
``` python
print("-------------------开始监视代码----------------------")
print(do_match)
print("-------------------结束监视代码----------------------")
exit()
if do_match:
    # Perform the matching.
    pred = matching({"image0": inp0, "image1": inp1})
```
结果为：
```
-------------------开始监视代码----------------------
True
-------------------结束监视代码----------------------
```
也就是说，接下来将会执行`if do_match:`语句包裹的代码。我们来完整地看一下`if do_match:`语句包裹的代码：
``` python
if do_match:
    # Perform the matching.
    pred = matching({"image0": inp0, "image1": inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    matches, conf = pred["matches0"], pred["matching_scores0"]
    timer.update("matcher")

    # Write the matches to disk.
    out_matches = {
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "matches": matches,
        "match_confidence": conf,
    }
    np.savez(str(matches_path), **out_matches)
```
这段代码可以说是整个推理脚本的真正核心。这段代码执行了最关键的特征点匹配的工作。因此，这段代码必须最为深入仔细地来研究，仔细弄懂它究竟是怎么使用论文作者提供的模型来进行特征点的提取和匹配的。

首先，这段特征点匹配的代码调用了之前初始化的模型，也就是下面的一行：
``` python
pred = matching({"image0": inp0, "image1": inp1})
```
上面已经测试过了，输入SuperGlue网络中的数据结构是两张匹配图像的`<class 'torch.Tensor'>`类型的数据结构，也就是`inp0`和`inp1`这两个变量。（看一下上面print出来的内容可知：`type(inp0)： <class 'torch.Tensor'>`，`type(inp1)： <class 'torch.Tensor'>`，`inp0.shape： torch.Size([1, 1, 900, 1600])`，`inp1.shape： torch.Size([1, 1, 900, 1600])`）
注意：这一行代码`pred = matching({"image0": inp0, "image1": inp1})`以及接下来的几行代码都是极为关键的，因为这一行代码是执行推理模型的最最核心的代码。因此，对我来说，最最重要的就是，弄清楚输入到推理模型中的数据的结构，以及从推理模型中输出出来的数据的结构。弄清楚了输入模型的数据和从模型输出的数据的结构，就弄清楚了该如何在我未来的任务中使用`SuperGlue`模型。因此，接下来的一些分析，应该是这份笔记中最最重要的分析，应该格外地仔细。
关于PyTorch模型的调用，不应该写出`forward`，相关的解释参见[这里](https://stackoverflow.com/questions/66594136/calling-the-forward-method-in-pytorch-vs-calling-the-model-instance)和[这里](https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput)。

输入到推理模型中的两张匹配图像的数据结构上面已经完全弄清楚了。下面来测试一下从推理模型中输出的数据结构。按照预期，从推理模型中输出的，应该是匹配的特征点对，以及这些匹配特征点对的匹配程度置信度。这些匹配的特征点对和相应的匹配置信度应该是被组织到了`pred`这个变量里。因为我已经测试了输出的`pred`是一个字典，因此接下来我们直接来看一下输出的`pred`字典中都有哪些键。测试下述的代码：
``` python
if do_match:
    # Perform the matching.
    pred = matching({"image0": inp0, "image1": inp1})
    print("-------------------开始监视代码----------------------")
    print(type(pred))
    print("-------------------我的分割线1----------------------")
    for i in pred.keys():
        print(i)
    print("-------------------结束监视代码----------------------")
    exit()
```
结果为：
```
-------------------开始监视代码----------------------
<class 'dict'>
-------------------我的分割线1----------------------
keypoints0
scores0
descriptors0
keypoints1
scores1
descriptors1
matches0
matches1
matching_scores0
matching_scores1
-------------------结束监视代码----------------------
```
（注意，我已经测试过了。`pred`这个字典的`key`都是字符串`<class 'str'>`类型的变量。）
可以看到，一对匹配图像对经过SuperGlue模型（也就是`matching`变量，这个`matching`变量里面包含了完整的SuperGlue模型）映射后，得到了一个名为`pred`的字典。按照预期，这个`pred`字典里面应该包含匹配的特征点对和每一个点对的匹配置信度。这个`pred`字典的全部键是`keypoints0`，`scores0`，`descriptors0`，`keypoints1`，`scores1`，`descriptors1`，`matches0`，`matches1`，`matching_scores0`，`matching_scores1`。接下来再把每个键对应的值的类型print出来看一下。运行下述代码：
``` python
if do_match:
    # Perform the matching.
    pred = matching({"image0": inp0, "image1": inp1})
    print("-------------------开始监视代码----------------------")
    print("type(pred)：", type(pred))
    print("-------------------我的分割线1----------------------")
    for i in pred.keys():
        print(f"字典pred的键 {i} 所对应的值的类型是：", type(pred[i]))
    print("-------------------结束监视代码----------------------")
    exit()
```
结果为：
```
-------------------开始监视代码----------------------
type(pred)： <class 'dict'>
-------------------我的分割线1----------------------
字典pred的键 keypoints0 所对应的值的类型是： <class 'list'>
字典pred的键 scores0 所对应的值的类型是： <class 'tuple'>
字典pred的键 descriptors0 所对应的值的类型是： <class 'list'>
字典pred的键 keypoints1 所对应的值的类型是： <class 'list'>
字典pred的键 scores1 所对应的值的类型是： <class 'tuple'>
字典pred的键 descriptors1 所对应的值的类型是： <class 'list'>
字典pred的键 matches0 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 matches1 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 matching_scores0 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 matching_scores1 所对应的值的类型是： <class 'torch.Tensor'>
-------------------结束监视代码----------------------
```
这样就看出来了这个`pred`字典的所有键对应的值的类型。最终我需要用到的究竟是哪个键的值，我暂时还不清楚，还需要深入研究下面的代码。

下面的一行代码`pred = {k: v[0].cpu().numpy() for k, v in pred.items()}`对字典`pred`做了一些处理。所做的处理是：字典`pred`中的所有的键都原封不动，但是值的数据类型全部被转换成了`numpy.ndarray`数据类型。我们来试运行下述代码：
``` python
if do_match:
    # Perform the matching.
    pred = matching({"image0": inp0, "image1": inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    print("-------------------开始监视代码----------------------")
    print("type(pred)：", type(pred))
    print("-------------------我的分割线1----------------------")
    for i in pred.keys():
        print(f"字典pred的键 {i} 所对应的值的类型是：", type(pred[i]))
    print("-------------------结束监视代码----------------------")
    exit()
```
结果为：
```
-------------------开始监视代码----------------------
type(pred)： <class 'dict'>
-------------------我的分割线1----------------------
字典pred的键 keypoints0 所对应的值的类型是： <class 'numpy.ndarray'>
字典pred的键 scores0 所对应的值的类型是： <class 'numpy.ndarray'>
字典pred的键 descriptors0 所对应的值的类型是： <class 'numpy.ndarray'>
字典pred的键 keypoints1 所对应的值的类型是： <class 'numpy.ndarray'>
字典pred的键 scores1 所对应的值的类型是： <class 'numpy.ndarray'>
字典pred的键 descriptors1 所对应的值的类型是： <class 'numpy.ndarray'>
字典pred的键 matches0 所对应的值的类型是： <class 'numpy.ndarray'>
字典pred的键 matches1 所对应的值的类型是： <class 'numpy.ndarray'>
字典pred的键 matching_scores0 所对应的值的类型是： <class 'numpy.ndarray'>
字典pred的键 matching_scores1 所对应的值的类型是： <class 'numpy.ndarray'>
-------------------结束监视代码----------------------
```
所以，字典`pred`的所有的值都被转化成了`<class 'numpy.ndarray'>`类型的数据。至于说，为什么要做这样的一步数据类型的转换，我也不知道。可能，以后等我看过更多的开源代码之后，我会知道到底什么时候该用哪种数据类型吧。

接下来，我要看看`pred`字典的每个键所对应的值具有什么形状。测试一下下述代码：
``` python
if do_match:
    # Perform the matching.
    pred = matching({"image0": inp0, "image1": inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    print("-------------------开始监视代码----------------------")
    print("type(pred)：", type(pred))
    print("-------------------我的分割线1----------------------")
    for i in pred.keys():
        print(f"pred[{i}].shape：", pred[i].shape)
    print("-------------------结束监视代码----------------------")
    exit()
```
结果为：
```
-------------------开始监视代码----------------------
type(pred)： <class 'dict'>
-------------------我的分割线1----------------------
pred[keypoints0].shape： (100, 2)
pred[scores0].shape： (100,)
pred[descriptors0].shape： (256, 100)
pred[keypoints1].shape： (100, 2)
pred[scores1].shape： (100,)
pred[descriptors1].shape： (256, 100)
pred[matches0].shape： (100,)
pred[matches1].shape： (100,)
pred[matching_scores0].shape： (100,)
pred[matching_scores1].shape： (100,)
-------------------结束监视代码----------------------
```
由于接下来的几行代码需要用到`pred`字典的`"keypoints0"`，`"keypoints1"`，`"matches0"`，`"matching_scores0"`这几个键的值，因此，我们再来把我们已经获得的关于这几个键的值的信息整理一下。试运行下述代码：
``` python
if do_match:
    # Perform the matching.
    pred = matching({"image0": inp0, "image1": inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    print("-------------------开始监视代码----------------------")
    print('type(pred["keypoints0"])：', type(pred["keypoints0"]))
    print('type(pred["keypoints1"])：', type(pred["keypoints1"]))
    print('type(pred["matches0"])：', type(pred["matches0"]))
    print('type(pred["matching_scores0"])：', type(pred["matching_scores0"]))
    print("-------------------我的分割线1----------------------")
    print('pred["keypoints0"].shape：', pred["keypoints0"].shape)
    print('pred["keypoints1"].shape：', pred["keypoints1"].shape)
    print('pred["matches0"].shape：', pred["matches0"].shape)
    print('pred["matching_scores0"].shape：', pred["matching_scores0"].shape)
    print("-------------------我的分割线2----------------------")
    print('pred["keypoints0"]：', pred["keypoints0"])
    print('pred["keypoints1"]：', pred["keypoints1"])
    print('pred["matches0"]：', pred["matches0"])
    print('pred["matching_scores0"]：', pred["matching_scores0"])
    print("-------------------结束监视代码----------------------")
    exit()
```
结果为：
```
-------------------开始监视代码----------------------
type(pred["keypoints0"])： <class 'numpy.ndarray'>
type(pred["keypoints1"])： <class 'numpy.ndarray'>
type(pred["matches0"])： <class 'numpy.ndarray'>
type(pred["matching_scores0"])： <class 'numpy.ndarray'>
-------------------我的分割线1----------------------
pred["keypoints0"].shape： (100, 2)
pred["keypoints1"].shape： (100, 2)
pred["matches0"].shape： (100,)
pred["matching_scores0"].shape： (100,)
-------------------我的分割线2----------------------
pred["keypoints0"]： [[1195.  255.]
 [1407.  532.]
 [ 792.  303.]
 [  20.  394.]
 [  21.  379.]
 [1264.  503.]
 [1392.  388.]
 [ 758.  392.]
 [1271.  484.]
 [ 628.  455.]
 [1502.  520.]
 [  36.  365.]
 [1218.  330.]
 [1346.  355.]
 [1193.  214.]
 [1487.  383.]
 [1493.   84.]
 [1061.  512.]
 [ 999.  491.]
 [1459.  124.]
 [1401.  370.]
 [ 131.  422.]
 [ 741.  494.]
 [1295.   52.]
 [ 357.  421.]
 [ 316.  165.]
 [1206.  398.]
 [1196.  451.]
 [1282.  466.]
 [ 513.  449.]
 [ 205.  442.]
 [1203.  322.]
 [ 917.  451.]
 [1198.  405.]
 [1259.  613.]
 [1018.  446.]
 [1354.  107.]
 [  15.  569.]
 [ 486.  644.]
 [1154.  265.]
 [ 726.  389.]
 [ 112.  518.]
 [ 570.  298.]
 [ 158.  419.]
 [ 774.  494.]
 [ 661.  445.]
 [ 245.  420.]
 [  65.  409.]
 [ 172.  454.]
 [ 117.  419.]
 [1484.   43.]
 [ 628.  443.]
 [1474.  521.]
 [  65.  395.]
 [ 110.  325.]
 [1505.   58.]
 [1379.  380.]
 [1176.  403.]
 [ 617.  458.]
 [ 906.  472.]
 [ 746.  407.]
 [1077.  391.]
 [ 222.  393.]
 [1142.  560.]
 [1128.  535.]
 [ 980.  443.]
 [1261.  559.]
 [1473.  386.]
 [  19.  409.]
 [ 206.  453.]
 [1236.  425.]
 [ 219.  453.]
 [1523.  151.]
 [1189.  385.]
 [1212.  373.]
 [  34.  395.]
 [1234.  325.]
 [  25.  532.]
 [  19.  424.]
 [1443.  556.]
 [1418.  453.]
 [ 188.  420.]
 [1513.  359.]
 [ 800.  313.]
 [ 383.  422.]
 [1211.  288.]
 [ 324.  423.]
 [ 560.  286.]
 [ 208.  167.]
 [ 509.  372.]
 [1521.  426.]
 [ 513.  424.]
 [ 851.  486.]
 [1204.  262.]
 [1439.  478.]
 [  45.  444.]
 [1135.  370.]
 [ 772.  451.]
 [1404.  563.]
 [1168.  373.]]
pred["keypoints1"]： [[ 150.  482.]
 [1223.  434.]
 [1389.   16.]
 [ 117.  428.]
 [ 163.  421.]
 [ 477.  455.]
 [ 207.  420.]
 [  73.  410.]
 [ 966.  379.]
 [ 807.  295.]
 [ 224.  418.]
 [ 164.  376.]
 [ 405.  163.]
 [1077.  242.]
 [ 912.  426.]
 [1515.  429.]
 [ 215.  474.]
 [ 163.  406.]
 [ 204.  341.]
 [ 177.  507.]
 [1466.   62.]
 [ 480.  439.]
 [  82.  475.]
 [1431.  390.]
 [ 832.  443.]
 [ 297.  375.]
 [1086.  291.]
 [ 224.  404.]
 [ 493.  439.]
 [  93.  386.]
 [  93.  373.]
 [ 164.  391.]
 [ 208.  407.]
 [1032.  274.]
 [1087.  305.]
 [1274.  394.]
 [ 271.  516.]
 [ 534.  321.]
 [1020.  284.]
 [ 225.  474.]
 [1471.  592.]
 [ 198.  441.]
 [1426.  533.]
 [ 318.  375.]
 [1211.   85.]
 [ 413.  456.]
 [1512.  507.]
 [ 134.  428.]
 [1368.  378.]
 [  23.  192.]
 [ 326.  515.]
 [ 201.  527.]
 [1363.  319.]
 [ 164.  491.]
 [1496.   37.]
 [1418.  424.]
 [ 502.  175.]
 [ 198.  225.]
 [1385.  349.]
 [ 579.  186.]
 [1299.  412.]
 [ 149.  393.]
 [ 104.  386.]
 [ 739.  257.]
 [1446.  490.]
 [1501.  494.]
 [ 226.  523.]
 [1047.  447.]
 [  96.  275.]
 [ 122.   68.]
 [ 318.  437.]
 [  92.  474.]
 [1098.  431.]
 [ 948.  471.]
 [1408.   30.]
 [1331.  508.]
 [ 149.  428.]
 [1336.  222.]
 [1520.  277.]
 [ 841.  414.]
 [ 117.  415.]
 [1203.  465.]
 [  80.  384.]
 [ 275.  462.]
 [1374.  320.]
 [ 151.  460.]
 [ 548.  445.]
 [ 753.  442.]
 [ 785.  276.]
 [ 224.  391.]
 [ 134.  441.]
 [1325.  186.]
 [ 226.  334.]
 [1109.  283.]
 [1308.  275.]
 [1470.  354.]
 [ 173.  480.]
 [ 412.  434.]
 [1380.  259.]
 [ 268.   94.]]
pred["matches0"]： [-1 65 -1 31 11 -1 -1 -1 -1 -1 -1 -1 52 -1 -1 -1 -1 -1 81 -1 -1 -1 73 74
 -1 59 48 -1 -1 -1 -1 -1 72 -1 -1 -1 20 -1 -1 -1  8 36  9 -1 -1 14 -1 27
 -1 -1 -1 79 -1 89 92 -1 -1 -1 24 67 -1 -1 -1 -1 75 -1 -1 -1 17 -1 -1 -1
 -1 -1 -1 -1 84 51  4 -1 15 -1 -1 -1 -1 -1 -1 88 12 -1 -1 -1 -1 -1 -1 41
 -1 -1 46 -1]
pred["matching_scores0"]： [0.13380201 0.9424425  0.03885581 0.91125965 0.9151573  0.01251809
 0.         0.         0.         0.         0.         0.
 0.9446146  0.05079027 0.         0.         0.         0.
 0.9507499  0.13739356 0.         0.         0.7629122  0.54263514
 0.         0.7709297  0.95441735 0.         0.02070532 0.
 0.19323339 0.         0.31125242 0.         0.         0.13291469
 0.4757099  0.         0.         0.         0.31050265 0.8418391
 0.88900244 0.         0.         0.43917206 0.         0.87394863
 0.14354663 0.         0.         0.23306607 0.         0.9421702
 0.20865494 0.         0.         0.07574096 0.5240165  0.23241605
 0.         0.         0.         0.         0.6805611  0.
 0.06329903 0.         0.925631   0.         0.         0.
 0.         0.         0.04509572 0.         0.6621486  0.65668684
 0.8835403  0.         0.94609725 0.01596557 0.03657727 0.
 0.03053425 0.         0.         0.8210559  0.88948035 0.
 0.         0.         0.         0.08950654 0.         0.94169813
 0.         0.         0.22881694 0.        ]
-------------------结束监视代码----------------------
```
注意：`pred["matches0"]`里的`-1`代表无效匹配。参见`/SuperGluePretrainedNetwork/models/superglue.py`代码中的`class SuperGlue(nn.Module): forward(self, data):`的代码。

至此，最最关键的特征点匹配的过程就算完成了。剩下的步骤就是保存数据了。接下来的代码如下：
``` python
kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
matches, conf = pred["matches0"], pred["matching_scores0"]
timer.update("matcher")

# Write the matches to disk.
out_matches = {
    "keypoints0": kpts0,
    "keypoints1": kpts1,
    "matches": matches,
    "match_confidence": conf,
}
np.savez(str(matches_path), **out_matches)
```
这些代码就是在保存数据。里面没有什么特别难的部分。我们只需要看一下`np.savez()`函数的用法以及`**dict`语法的用法即可。关于Python的`**dict`语法，参见[这里的解释](https://stackoverflow.com/questions/21809112/what-does-tuple-and-dict-mean-in-python)。试运行下述代码：
``` python
# Write the matches to disk.
out_matches = {
    "keypoints0": kpts0,
    "keypoints1": kpts1,
    "matches": matches,
    "match_confidence": conf,
}
print("-------------------开始监视代码----------------------")
print("str(matches_path)：", str(matches_path))
print("-------------------结束监视代码----------------------")
exit()
np.savez(str(matches_path), **out_matches)
```
结果为：
```
-------------------开始监视代码----------------------
str(matches_path)： dump_match_pairs_outdoor/photo1_1_photo1_2_matches.npz
-------------------结束监视代码----------------------
```
关于`np.savez()`的用法，参见[numpy.savez官方文档](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)

接下来的几行代码的功能是：去掉不合法的匹配点对：
``` python
# Keep the matching keypoints.
valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]
```
运行如下的测试代码：
``` python
# Keep the matching keypoints.
valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]
print("-------------------开始监视代码----------------------")
print("type(valid)：", type(valid))
print("type(mkpts0)：", type(mkpts0))
print("type(mkpts1)：", type(mkpts1))
print("type(mconf)：", type(mconf))
print("-------------------我的分割线1----------------------")
print("valid.shape：", valid.shape)
print("mkpts0.shape：", mkpts0.shape)
print("mkpts1.shape：", mkpts1.shape)
print("mconf.shape：", mconf.shape)
print("-------------------我的分割线2----------------------")
print("valid：", valid)
print("mkpts0：", mkpts0)
print("mkpts1：", mkpts1)
print("mconf：", mconf)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
type(valid)： <class 'numpy.ndarray'>
type(mkpts0)： <class 'numpy.ndarray'>
type(mkpts1)： <class 'numpy.ndarray'>
type(mconf)： <class 'numpy.ndarray'>
-------------------我的分割线1----------------------
valid.shape： (100,)
mkpts0.shape： (31, 2)
mkpts1.shape： (31, 2)
mconf.shape： (31,)
-------------------我的分割线2----------------------
valid： [False  True False  True  True False False False False False False False
  True False False False False False  True False False False  True  True
 False  True  True False False False False False  True False False False
  True False False False  True  True  True False False  True False  True
 False False False  True False  True  True False False False  True  True
 False False False False  True False False False  True False False False
 False False False False  True  True  True False  True False False False
 False False False  True  True False False False False False False  True
 False False  True False]
mkpts0： [[1407.  532.]
 [  20.  394.]
 [  21.  379.]
 [1218.  330.]
 [ 999.  491.]
 [ 741.  494.]
 [1295.   52.]
 [ 316.  165.]
 [1206.  398.]
 [ 917.  451.]
 [1354.  107.]
 [ 726.  389.]
 [ 112.  518.]
 [ 570.  298.]
 [ 661.  445.]
 [  65.  409.]
 [ 628.  443.]
 [  65.  395.]
 [ 110.  325.]
 [ 617.  458.]
 [ 906.  472.]
 [1128.  535.]
 [  19.  409.]
 [1234.  325.]
 [  25.  532.]
 [  19.  424.]
 [1418.  453.]
 [ 560.  286.]
 [ 208.  167.]
 [  45.  444.]
 [1404.  563.]]
mkpts1： [[1501.  494.]
 [ 164.  391.]
 [ 164.  376.]
 [1363.  319.]
 [1203.  465.]
 [ 948.  471.]
 [1408.   30.]
 [ 579.  186.]
 [1368.  378.]
 [1098.  431.]
 [1466.   62.]
 [ 966.  379.]
 [ 271.  516.]
 [ 807.  295.]
 [ 912.  426.]
 [ 224.  404.]
 [ 841.  414.]
 [ 224.  391.]
 [ 226.  334.]
 [ 832.  443.]
 [1047.  447.]
 [1331.  508.]
 [ 163.  406.]
 [1374.  320.]
 [ 201.  527.]
 [ 163.  421.]
 [1515.  429.]
 [ 785.  276.]
 [ 405.  163.]
 [ 198.  441.]
 [1512.  507.]]
mconf： [0.9424425  0.91125965 0.9151573  0.9446146  0.9507499  0.7629122
 0.54263514 0.7709297  0.95441735 0.31125242 0.4757099  0.31050265
 0.8418391  0.88900244 0.43917206 0.87394863 0.23306607 0.9421702
 0.20865494 0.5240165  0.23241605 0.6805611  0.925631   0.6621486
 0.65668684 0.8835403  0.94609725 0.8210559  0.88948035 0.94169813
 0.22881694]
-------------------结束监视代码----------------------
```
由此知，这一对图像只有31对特征点成功匹配。最大特征点对的个数设置为了100对。通过这里的测试也可以知道：**要想获得一共有多少对特征点成功匹配，只需要看一看`mconf.shape`即可**。

接下来试运行下述代码：
``` python
print("-------------------开始监视代码----------------------")
print(do_eval)
print("-------------------结束监视代码----------------------")
exit()
if do_eval:
    # Estimate the pose and compute the pose error.
    assert len(pair) == 38, "Pair does not have ground truth info"
```
结果为：
```
-------------------开始监视代码----------------------
False
-------------------结束监视代码----------------------
```
由此知，`if do_eval:`语句包裹的代码在本次运行中不会被运行。即：我这次的运行，不会执行评估过程，只会执行可视化过程。

下面的可视化部分的代码，我暂时先不深入研究了。到此为止，这个推理脚本可以算是学习完毕了。总结一下，从这个推理脚本的学习中，我发现，在使用一个训练好的网络模型进行推理时，一定要搞清楚三点：
```
1.网络模型是如何加载已经训练好的权重的？
2.待评估的数据是如何被转化成网络可以用的数据格式的？网络可以用的是哪种数据格式？
3.网络输出的是哪种数据格式？网络输出的结果如何保存？
```
这三点是非常非常重要的。可以说，一个推理脚本的核心，就是要实现这三点的功能。所以，以后我自己写推理脚本时，也要特别注意，这三点的功能都是被如何实现的。