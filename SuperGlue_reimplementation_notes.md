# SuperGlue训练代码复现笔记

在这份笔记中，我将记录我自己试图复现SuperGlue训练代码的详细步骤。

## 配置相关环境
1.使用如下的命令安装conda虚拟环境：
``` bash
conda create --name superglue python=3.8
```

2.然后从Github上克隆[SuperGlue官方模型和评测脚本](https://github.com/magicleap/SuperGluePretrainedNetwork)（或者从我的Gitee上克隆[相应的镜像仓库](https://gitee.com/Jazihars/SuperGluePretrainedNetwork)）

3.使用下述命令安装相应依赖包（注意：一定要先运行`conda activate superglue`命令）：
``` bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install matplotlib
pip install black
pip install tqdm
pip install scipy
```

4.运行`pip list`命令，查看现有的依赖包为：
```
Package           Version
----------------- -----------
cycler            0.11.0
fonttools         4.28.5
kiwisolver        1.3.2
matplotlib        3.5.1
numpy             1.21.5
opencv-python     4.5.4.60
packaging         21.3
Pillow            8.4.0
pip               21.3.1
pyparsing         3.0.6
python-dateutil   2.8.2
setuptools        60.1.0
six               1.16.0
torch             1.8.1+cu111
torchvision       0.9.1+cu111
typing_extensions 4.0.1
wheel             0.37.1
```

## 配置训练数据集
使用下述命令将训练数据上传开发机：
``` bash
scp ...
```
然后在home目录下新建`coco2014`这个目录，将上传的训练数据压缩文件移入`coco2014`这个目录，运行`unzip train2014.zip`。我的数据目录是：`/home/users/XXX/coco2014/train2014`。

## 尝试自己复现SuperGlue的训练代码
由于[已有的SuperGlue开源PyTorch复现](https://github.com/HeatherJiaZG/SuperGlue-pytorch)似乎并不在活跃维护中，而且训练出来的模型无法和官方评测脚本兼容，因此尝试自己实现SuperGlue的PyTorch训练代码。

为了最大限度地应用已有的开源代码，我决定在[官方SuperGlue模型和评测脚本](https://github.com/magicleap/SuperGluePretrainedNetwork)的基础上，完全参考[已有的SuperGlue开源PyTorch复现](https://github.com/HeatherJiaZG/SuperGlue-pytorch)，将缺少的代码补全，然后再逐步修复bug的做法。这是我第一次尝试复现训练脚本，我并不确定这种策略一定能成功。

1.将[已有的SuperGlue开源PyTorch复现](https://github.com/HeatherJiaZG/SuperGlue-pytorch)中的`train.py`，`matchingForTraining.py`和`load_data.py`三份脚本完全复制过来，路径为：`/SuperGluePretrainedNetwork/train.py`，`/SuperGluePretrainedNetwork/models/matchingForTraining.py`和`/SuperGluePretrainedNetwork/load_data.py`。
2.在训练脚本`/SuperGluePretrainedNetwork/train.py`中，将训练路径参数`train_path`的默认值修改为：`/home/users/XXX/coco2014/train2014/`。
3.将[已有的SuperGlue开源PyTorch复现](https://github.com/HeatherJiaZG/SuperGlue-pytorch)中的`/SuperGlue-pytorch/models/utils.py`中的`def read_image_modified(image, resize, resize_float):`函数的代码复制到`/SuperGluePretrainedNetwork/models/utils.py`里面。
4.修复`AttributeError: module 'cv2' has no attribute 'xfeatures2d'`的错误：由于我安装的是最新版本的`opencv-python`，因此，直接使用`cv2.SIFT_create()`即可。

这份笔记暂时不更新了。因为我暂时不知道怎么解决后续的bug。