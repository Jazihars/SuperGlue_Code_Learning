# SuperGlue第三方训练代码学习笔记

在这个笔记中，我将对SuperGlue第三方训练代码进行梳理。弄清楚这个第三方SuperGlue训练代码的结构和功能。代码地址参见[这里](https://github.com/HeatherJiaZG/SuperGlue-pytorch)

## 环境配置
我在Linux终端里，运行下述查看CentOS系统版本的命令：
``` bash
cat /etc/redhat-release
```
可以看到如下输出：
```
CentOS Linux release 7.5.1804 (Core)
```
这就明确了：我的接下来的实验都是在`CentOS 7.5.1804`版本的`Linux`系统上进行的。

首先选定一个当前目录，然后进入[Gitee上的SuperGlue-pytorch仓库](https://gitee.com/Jazihars/SuperGlue-pytorch)，按照如下的命令，克隆仓库：
``` bash
git clone https://gitee.com/Jazihars/SuperGlue-pytorch.git
cd SuperGlue-pytorch/
```

然后按照如下的命令配置conda虚拟环境：
``` bash
conda create -n superglue python=3.8 -y
conda activate superglue
```

使用下述命令，测试一下我的CUDA版本：
``` bash
nvcc -V
```
结果为：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Tue_Sep_15_19:10:02_PDT_2020
Cuda compilation tools, release 11.1, V11.1.74
Build cuda_11.1.TC455_06.29069683_0
```
由此知，我的CUDA版本是11.1

进入`/SuperGlue-pytorch/requirements.txt`文件，将这个文件的内容改为下述内容：
```
matplotlib>=3.1.3
torch>=1.1.0
opencv-python
numpy>=1.18.1
```

接下来执行下述命令，安装依赖环境：
``` bash
pip install -r requirements.txt
```

用下述命令，安装代码格式化工具`black`：
``` bash
pip install black
```

在`/SuperGlue-pytorch/train.py`代码里，将我的`coco2014`数据的路径写入`train_path`参数的默认值（注意，最后一定要有一个`/`来结尾）：
``` python
parser.add_argument(
    "--train_path",
    type=str,
    default="/mnt/data-2/data/zitong.yin/coco2014/train2014/",
    help="Path to the directory of training imgs.",
)
```

使用下述命令，继续安装缺少的依赖包：
``` bash
pip install scipy
pip install tqdm
```

进入`/SuperGlue-pytorch/load_data.py`代码里，将`self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)`这句话改为`self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)`。修改后的代码如下：
``` python
class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, nfeatures):

        self.files = []
        self.files += [train_path + f for f in os.listdir(train_path)]

        self.nfeatures = nfeatures
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
```

至此，[SuperGlue第三方训练代码](https://github.com/HeatherJiaZG/SuperGlue-pytorch)的环境准备工作就算完成了。下面可以进入正式的训练调试阶段了。


## 训练调试
首先，因为我们需要先确定代码能够跑通，所以我先设置一个比较小的`epoch`来进行接下来的调试。在`/SuperGlue-pytorch/train.py`代码里，修改`epoch`参数的默认值：
``` python
parser.add_argument("--epoch", type=int, default=2, help="Number of epoches")
```
我先设置2个epoch来测试一下代码到底能不能跑通。

在`/SuperGlue-pytorch/`目录下，运行下述命令：
``` bash
python train.py
```
看到代码跑通了。但跑通之后，训练了一会，终端中输出了如下的报错：
```
(/mnt/data-2/data/zitong.yin/conda_env/superglue) [zitong.yin@gpu-dev020 SuperGlue-pytorch]$ python train.py 
Namespace(batch_size=1, cache=False, epoch=2, eval=False, eval_input_dir='assets/scannet_sample_images/', eval_output_dir='dump_match_pairs/', eval_pairs_list='assets/scannet_sample_pairs_with_gt.txt', fast_viz=False, keypoint_threshold=0.005, learning_rate=0.0001, match_threshold=0.2, max_keypoints=1024, max_length=-1, nms_radius=4, opencv_display=False, resize=[640, 480], resize_float=False, show_keypoints=False, shuffle=False, sinkhorn_iterations=20, superglue='indoor', train_path='/mnt/data-2/data/zitong.yin/coco2014/train2014/', viz=False, viz_extension='png')
Will write visualization images to directory "dump_match_pairs"
Epoch [1/2], Step [50/82783], Loss: 1.5466
Epoch [1/2], Step [100/82783], Loss: 1.3900
Epoch [1/2], Step [150/82783], Loss: 1.2706
Epoch [1/2], Step [200/82783], Loss: 1.1972
Epoch [1/2], Step [250/82783], Loss: 1.2834
Epoch [1/2], Step [300/82783], Loss: 1.3147
Epoch [1/2], Step [350/82783], Loss: 1.3112
Traceback (most recent call last):
  File "train.py", line 228, in <module>
    data = superglue(pred)
  File "/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/data-2/data/zitong.yin/SuperGlue-pytorch/models/superglue.py", line 260, in forward
    desc1 = desc1 + self.kenc(kpts1, torch.transpose(data['scores1'], 0, 1))
  File "/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/data-2/data/zitong.yin/SuperGlue-pytorch/models/superglue.py", line 83, in forward
    return self.encoder(torch.cat(inputs, dim=1))
  File "/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/modules/container.py", line 141, in forward
    input = module(input)
  File "/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/modules/instancenorm.py", line 57, in forward
    return F.instance_norm(
  File "/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/functional.py", line 2326, in instance_norm
    _verify_spatial_size(input.size())
  File "/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/functional.py", line 2293, in _verify_spatial_size
    raise ValueError("Expected more than 1 spatial element when training, got input size {}".format(size))
ValueError: Expected more than 1 spatial element when training, got input size torch.Size([1, 32, 1])
```
（注意：在我的多次测试中，我发现，这个错误的出现有些不确定。在上面的这次测试中，是训练了350步后报了这个错，但我之前测试的时候也有训练了一千多步以后报了这个错的情况。）
这个错误我不知道该怎么处理，因此先去Github上看看有没有别人也遇到了同样的报错。在[SuperGlue-pytorch的问题页面](https://github.com/HeatherJiaZG/SuperGlue-pytorch/issues)里，发现有人问了[同样的问题](https://github.com/HeatherJiaZG/SuperGlue-pytorch/issues/17)。接下来我先尝试一下这个问题下面有人提到的一种解决方案。注意：此时我并不确定这个人提供的解决方案就一定能行，所以我现在是在尝试中。进入`SuperGlue-pytorch/load_data.py`代码，再找到`class SparseDataset(Dataset):`类下面的`def __getitem__(self, idx):`函数，将下面的这行代码改为如下的形式：
``` python
#原始代码：
if len(kp1) < 1 or len(kp2) < 1:
    return {
        "keypoints0": torch.zeros([0, 0, 2], dtype=torch.double),
        "keypoints1": torch.zeros([0, 0, 2], dtype=torch.double),
        "descriptors0": torch.zeros([0, 2], dtype=torch.double),
        "descriptors1": torch.zeros([0, 2], dtype=torch.double),
        "image0": image,
        "image1": warped,
        "file_name": file_name,
    }


#修改后的代码
if len(kp1) <= 1 or len(kp2) <= 1:
    return {
        "keypoints0": torch.zeros([0, 0, 2], dtype=torch.double),
        "keypoints1": torch.zeros([0, 0, 2], dtype=torch.double),
        "descriptors0": torch.zeros([0, 2], dtype=torch.double),
        "descriptors1": torch.zeros([0, 2], dtype=torch.double),
        "image0": image,
        "image1": warped,
        "file_name": file_name,
    }
```

接下来测试一下完整地训练一个epoch。为了防止关闭vscode而导致训练终止，我们来创建一个tmux终端，在tmux终端里面完整地训练1个epoch。先取消激活conda虚拟环境，使用下述命令建立tmux终端：
``` bash
conda deactivate
tmux new -s superglue
```
然后在新建立的tmux终端里执行下述命令开始训练：
``` bash
conda activate superglue
CUDA_VISIBLE_DEVICES=3 python train.py
```
接下来就可以按`Ctrl+b，d`来暂时退出名为superglue的tmux终端了。如果要重新回到这个tmux终端里，只需执行`tmux attach -t superglue`即可。

下面我们先退出tmux终端，让tmux终端里的训练继续进行。我们现在来逐行分析一下`/SuperGlue-pytorch/train.py`代码。

## 对训练脚本的逐行分析
### 最开始的导入模块和初始化参数
我在训练时，采用的是单机单卡训练（目前暂且先单机单卡，以后再研究如何单机多卡或多机多卡）。在`/SuperGlue-pytorch/`目录下，执行如下的训练命令（前提是激活`superglue`conda虚拟环境）：
``` bash
python train.py
```
就可以开始训练了。

我们还是逐行来分析`/SuperGlue-pytorch/train.py`代码。仔细地搞懂里面的每一行的用法。
`/SuperGlue-pytorch/train.py`代码最开头是导入模块的部分。这些部分以后我们用到了相应的模块的时候再来分析。在导入模块的部分之后，是初始化参数的部分。初始化参数的部分的代码如下：
``` python
parser = argparse.ArgumentParser(
    description="Image pair matching and pose evaluation with SuperGlue",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    "--fast_viz",
    action="store_true",
    help="Use faster image visualization based on OpenCV instead of Matplotlib",
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
    "--eval_pairs_list",
    type=str,
    default="assets/scannet_sample_pairs_with_gt.txt",
    help="Path to the list of image pairs for evaluation",
)
parser.add_argument(
    "--shuffle", action="store_true", help="Shuffle ordering of pairs before processing"
)
parser.add_argument(
    "--max_length", type=int, default=-1, help="Maximum number of pairs to evaluate"
)

parser.add_argument(
    "--eval_input_dir",
    type=str,
    default="assets/scannet_sample_images/",
    help="Path to the directory that contains the images",
)
parser.add_argument(
    "--eval_output_dir",
    type=str,
    default="dump_match_pairs/",
    help="Path to the directory in which the .npz results and optional,"
    "visualizations are written",
)
parser.add_argument("--learning_rate", type=int, default=0.0001, help="Learning rate")

parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
parser.add_argument(
    "--train_path",
    type=str,
    default="/mnt/data-2/data/zitong.yin/coco2014/train2014/",
    help="Path to the directory of training imgs.",
)
parser.add_argument("--epoch", type=int, default=1, help="Number of epoches")
```
对于这些初始化参数部分的代码，我们其实没有什么好说的。这些代码就是固定的用法，只需要记住就行了。以后如果我要编写自己的网络，也是完全一样地模仿着这些初始化参数的代码来初始化我自己的参数。（或者，参考[Swin Transformer的初始化参数的代码](https://github.com/Jazihars/Swin_Transformer_Code_Learning/blob/main/Swin_Transformer_Code_Study_Notes.md#%E8%AE%AD%E7%BB%83%E5%85%A5%E5%8F%A3)。）不过，我们重点需要留意一下，这些初始化好的参数是储存在什么样的数据结构里，以及之后该如何调用这些参数。带着这个问题，我们进入下面的主程序训练入口的部分。

### 正式进入训练入口if __name__ == "__main__":
主程序的正式训练入口的代码如下：
``` python
if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)

    # make sure the flags are properly used
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

    # store viz results
    eval_output_dir = Path(opt.eval_output_dir)
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    print(
        "Will write visualization images to", 'directory "{}"'.format(eval_output_dir)
    )
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

    # load training data
    train_set = SparseDataset(opt.train_path, opt.max_keypoints)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True
    )

    superglue = SuperGlue(config.get("superglue", {}))

    if torch.cuda.is_available():
        superglue.cuda()  # make sure it trains on GPU
    else:
        print("### CUDA not available ###")
    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    mean_loss = []

    # start training
    for epoch in range(1, opt.epoch + 1):
        epoch_loss = 0
        superglue.double().train()
        for i, pred in enumerate(train_loader):
            for k in pred:
                if k != "file_name" and k != "image0" and k != "image1":
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].cuda())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).cuda())

            data = superglue(pred)
            for k, v in pred.items():
                pred[k] = v[0]
            pred = {**pred, **data}

            if pred["skip_train"] == True:  # image has no keypoint
                continue

            # process loss
            Loss = pred["loss"]
            epoch_loss += Loss.item()
            mean_loss.append(Loss)

            superglue.zero_grad()
            Loss.backward()
            optimizer.step()

            # for every 50 images, print progress and visualize the matches
            if (i + 1) % 50 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch,
                        opt.epoch,
                        i + 1,
                        len(train_loader),
                        torch.mean(torch.stack(mean_loss)).item(),
                    )
                )
                mean_loss = []

                ### eval ###
                # Visualize the matches.
                superglue.eval()
                image0, image1 = (
                    pred["image0"].cpu().numpy()[0] * 255.0,
                    pred["image1"].cpu().numpy()[0] * 255.0,
                )
                kpts0, kpts1 = (
                    pred["keypoints0"].cpu().numpy()[0],
                    pred["keypoints1"].cpu().numpy()[0],
                )
                matches, conf = (
                    pred["matches0"].cpu().detach().numpy(),
                    pred["matching_scores0"].cpu().detach().numpy(),
                )
                image0 = read_image_modified(image0, opt.resize, opt.resize_float)
                image1 = read_image_modified(image1, opt.resize, opt.resize_float)
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                viz_path = eval_output_dir / "{}_matches.{}".format(
                    str(i), opt.viz_extension
                )
                color = cm.jet(mconf)
                stem = pred["file_name"]
                text = []

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
                    stem,
                    stem,
                    opt.show_keypoints,
                    opt.fast_viz,
                    opt.opencv_display,
                    "Matches",
                )

            # process checkpoint for every 5e3 images
            if (i + 1) % 5e3 == 0:
                model_out_path = "model_epoch_{}.pth".format(epoch)
                torch.save(superglue, model_out_path)
                print(
                    "Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}".format(
                        epoch, opt.epoch, i + 1, len(train_loader), model_out_path
                    )
                )

        # save checkpoint when an epoch finishes
        epoch_loss /= len(train_loader)
        model_out_path = "model_epoch_{}.pth".format(epoch)
        torch.save(superglue, model_out_path)
        print(
            "Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}".format(
                epoch, opt.epoch, epoch_loss, model_out_path
            )
        )
```
这段代码完成了从最开始的加载参数到最后的训练完成的整个过程。我们来一行一行地分析。
首先，最开始的两行代码如下：
``` python
opt = parser.parse_args()
print(opt)
```
这两行代码调用了之前的加载参数的部分，并为以后使用参数做了相应的准备。我们来测试一下这个调用参数的对象`opt`。试运行下述代码：
``` python
opt = parser.parse_args()
print("----------------------开始监视代码----------------------")
print("type(opt)：", type(opt))
print("----------------------我的分割线1----------------------")
print("opt：", opt)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(opt)： <class 'argparse.Namespace'>
----------------------我的分割线1----------------------
opt： Namespace(batch_size=1, cache=False, epoch=1, eval=False, eval_input_dir='assets/scannet_sample_images/', eval_output_dir='dump_match_pairs/', eval_pairs_list='assets/scannet_sample_pairs_with_gt.txt', fast_viz=False, keypoint_threshold=0.005, learning_rate=0.0001, match_threshold=0.2, max_keypoints=1024, max_length=-1, nms_radius=4, opencv_display=False, resize=[640, 480], resize_float=False, show_keypoints=False, shuffle=False, sinkhorn_iterations=20, superglue='indoor', train_path='/mnt/data-2/data/zitong.yin/coco2014/train2014/', viz=False, viz_extension='png')
----------------------结束监视代码----------------------
```
这个`opt`变量就是以后用来使用参数的变量。还是使用点语法`.`来调用存储在`opt`变量中的参数。

下面的四行是用来检查参数初始化是否合法的代码：
``` python
# make sure the flags are properly used
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
这四行代码也没有什么好说的，只是最简单的`assert`断言的用法，如果参数初始化不合法，会直接报错（意即，`assert`后面的语句如果为`False`，会引发`AssertionError`）。

下面的几行代码设定了可视化结果的保存路径：
``` python
# store viz results
eval_output_dir = Path(opt.eval_output_dir)
eval_output_dir.mkdir(exist_ok=True, parents=True)
print(
    "Will write visualization images to", 'directory "{}"'.format(eval_output_dir)
)
```
这些代码其实也没有什么好说的，就是一些固定的用法，只需要记住就行了。以后如果要写入我自己的可视化结果到某个路径，也是用这些代码。我们来测试一下下述代码：
``` python
# store viz results
eval_output_dir = Path(opt.eval_output_dir)
print("----------------------开始监视代码----------------------")
print("type(eval_output_dir)：", type(eval_output_dir))
print("----------------------我的分割线1----------------------")
print("eval_output_dir：", eval_output_dir)
print("----------------------结束监视代码----------------------")
exit()
eval_output_dir.mkdir(exist_ok=True, parents=True)
print(
    "Will write visualization images to", 'directory "{}"'.format(eval_output_dir)
)
```
结果为：
```
----------------------开始监视代码----------------------
type(eval_output_dir)： <class 'pathlib.PosixPath'>
----------------------我的分割线1----------------------
eval_output_dir： dump_match_pairs
----------------------结束监视代码----------------------
```
由此就明白了：可视化的结果会被保存在`/SuperGlue-pytorch/dump_match_pairs/`路径里。下面的一行代码`eval_output_dir.mkdir(exist_ok=True, parents=True)`就是创建这个文件夹的代码，之后打印出一行说明文字。

下面的一行代码初始化了一个参数变量，这个参数变量将用来调用之后要用到的一些关键参数：
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
```
我们来测试一下下面的代码：
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
print("----------------------开始监视代码----------------------")
print("type(config)：", type(config))
print("----------------------我的分割线1----------------------")
print("config：", config)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(config)： <class 'dict'>
----------------------我的分割线1----------------------
config： {'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024}, 'superglue': {'weights': 'indoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}}
----------------------结束监视代码----------------------
```
这个`config`字典是用来封装`superpoint`和`superglue`的一些关键参数的。之后要使用这些比较关键的参数，就会直接用这个`config`变量，而不会用`opt`变量。

接下来的两行代码是在加载训练数据：
``` python
# load training data
train_set = SparseDataset(opt.train_path, opt.max_keypoints)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True
)
```
我们可以看到，这两行代码，第一行是初始化一个训练数据集对象`train_set`，第二行是初始化一个训练数据加载器对象`train_loader`。而且，由于这里没有使用[torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)，因此SuperGlue-pytorch的训练会采用单机单卡的方式进行。关于如何实现单机多卡或多机多卡，可以参考一些比较优秀的开源代码（比如[Swin Transformer的代码](https://github.com/Jazihars/Swin_Transformer_Code_Learning/blob/main/Swin_Transformer_Code_Study_Notes.md)）。以后，我会考虑详细地学一下该如何实现单机多卡和多机多卡的编写。

我们先来进行一下下面的测试：
``` python
# load training data
train_set = SparseDataset(opt.train_path, opt.max_keypoints)
print("----------------------开始监视代码----------------------")
print("type(train_set)：", type(train_set))
print("----------------------我的分割线1----------------------")
print("len(train_set)：", len(train_set))
print("----------------------我的分割线2----------------------")
print("train_set：", train_set)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(train_set)： <class 'load_data.SparseDataset'>
----------------------我的分割线1----------------------
len(train_set)： 82783
----------------------我的分割线2----------------------
train_set： <load_data.SparseDataset object at 0x7fec02f51a30>
----------------------结束监视代码----------------------
```
`train_set`对象是一个储存了训练集的变量。我这次训练使用的是[coco2014训练数据集](http://images.cocodataset.org/zips/train2014.zip)。这个数据集在我的`opt.train_path`变量里。如果进入到`opt.train_path`这个路径里面再`ls`，会看到下面的内容：
```
COCO_train2014_000000193867.jpg  COCO_train2014_000000387595.jpg  COCO_train2014_000000581835.jpg
COCO_train2014_000000193878.jpg  COCO_train2014_000000387597.jpg  COCO_train2014_000000581839.jpg
COCO_train2014_000000193879.jpg  COCO_train2014_000000387598.jpg  COCO_train2014_000000581857.jpg
COCO_train2014_000000193880.jpg  COCO_train2014_000000387599.jpg  COCO_train2014_000000581860.jpg
COCO_train2014_000000193892.jpg  COCO_train2014_000000387601.jpg  COCO_train2014_000000581873.jpg
COCO_train2014_000000193901.jpg  COCO_train2014_000000387603.jpg  COCO_train2014_000000581880.jpg
COCO_train2014_000000193902.jpg  COCO_train2014_000000387604.jpg  COCO_train2014_000000581881.jpg
COCO_train2014_000000193923.jpg  COCO_train2014_000000387605.jpg  COCO_train2014_000000581882.jpg
COCO_train2014_000000193925.jpg  COCO_train2014_000000387606.jpg  COCO_train2014_000000581884.jpg
COCO_train2014_000000193931.jpg  COCO_train2014_000000387615.jpg  COCO_train2014_000000581900.jpg
COCO_train2014_000000193943.jpg  COCO_train2014_000000387616.jpg  COCO_train2014_000000581903.jpg
COCO_train2014_000000193947.jpg  COCO_train2014_000000387635.jpg  COCO_train2014_000000581904.jpg
COCO_train2014_000000193951.jpg  COCO_train2014_000000387666.jpg  COCO_train2014_000000581906.jpg
COCO_train2014_000000193953.jpg  COCO_train2014_000000387669.jpg  COCO_train2014_000000581909.jpg
COCO_train2014_000000193954.jpg  COCO_train2014_000000387672.jpg  COCO_train2014_000000581921.jpg
COCO_train2014_000000193972.jpg  COCO_train2014_000000387676.jpg
COCO_train2014_000000193977.jpg  COCO_train2014_000000387678.jpg
```
（注：这些只是一部分内容，更多内容终端里显示不下。）
我们再进入到`opt.train_path`路径`/mnt/data-2/data/zitong.yin/coco2014/train2014/`下，运行命令`ls -l | grep "^-" | wc -l`，结果为`82783`。这说明，训练数据有`82783`张无标签图片。
由此我们就明白了，SuperGlue-pytorch的训练数据集构造只有一行代码`train_set = SparseDataset(opt.train_path, opt.max_keypoints)`，这行代码接收一个数据集路径和一个最大关键点数为参数，输出一个训练数据集类`<class 'load_data.SparseDataset'>`对象。而且这行代码接收的数据集路径里面装的都是没有标签的图片。（至少到目前，我还不清楚这个SuperGlue-pytorch的训练代码是如何读取数据的？到底用没用数据标签？如果数据有标签，那究竟是在哪里读取标签的？这些问题之后我会深入地研究一下。）我现在先不展开分析数据集类`<class 'load_data.SparseDataset'>`的实现方法，之后再详细地分析。

下面再来看看DataLoader。我们可以看到，DataLoader的初始化，最主要的部分就是提供一个Dataset对象。所以其实，Dataset对象的编写才是重中之重。以后，一定要展开分析一下`<class 'load_data.SparseDataset'>`的代码。我们进行如下的测试，看一看训练用DataLoader对象究竟长什么样：
``` python
# load training data
train_set = SparseDataset(opt.train_path, opt.max_keypoints)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True
)
print("----------------------开始监视代码----------------------")
print("type(train_loader)：", type(train_loader))
print("----------------------我的分割线1----------------------")
print("len(train_loader)：", len(train_loader))
print("----------------------我的分割线2----------------------")
print("train_loader：", train_loader)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(train_loader)： <class 'torch.utils.data.dataloader.DataLoader'>
----------------------我的分割线1----------------------
len(train_loader)： 82783
----------------------我的分割线2----------------------
train_loader： <torch.utils.data.dataloader.DataLoader object at 0x7fe11fb47af0>
----------------------结束监视代码----------------------
```
由于`<class 'torch.utils.data.dataloader.DataLoader'>`类实现了`__iter__`方法，因此DataLoader对象是一个可迭代对象。我们试着迭代一下DataLoader对象看看效果。测试下述的代码：
``` python
# load training data
train_set = SparseDataset(opt.train_path, opt.max_keypoints)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True
)
print("----------------------开始监视代码----------------------")
print("type(train_loader)：", type(train_loader))
print("----------------------我的分割线1----------------------")
print("len(train_loader)：", len(train_loader))
print("----------------------我的分割线2----------------------")
print("train_loader：", train_loader)
print("----------------------我的分割线3----------------------")
temp = 1
for elementinloader in train_loader:
    print(f"train_loader里的第{temp}个元素的类型：", type(elementinloader))
    for key_key in elementinloader.keys():
        print(
            f"train_loader里的第{temp}个元素的键 {key_key} 所对应的值的类型是：{type(elementinloader[key_key])}"
        )
    temp += 1
    if temp == 2:
        break
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(train_loader)： <class 'torch.utils.data.dataloader.DataLoader'>
----------------------我的分割线1----------------------
len(train_loader)： 82783
----------------------我的分割线2----------------------
train_loader： <torch.utils.data.dataloader.DataLoader object at 0x7fb599dcfaf0>
----------------------我的分割线3----------------------
train_loader里的第1个元素的类型： <class 'dict'>
train_loader里的第1个元素的键 keypoints0 所对应的值的类型是：<class 'list'>
train_loader里的第1个元素的键 keypoints1 所对应的值的类型是：<class 'list'>
train_loader里的第1个元素的键 descriptors0 所对应的值的类型是：<class 'list'>
train_loader里的第1个元素的键 descriptors1 所对应的值的类型是：<class 'list'>
train_loader里的第1个元素的键 scores0 所对应的值的类型是：<class 'list'>
train_loader里的第1个元素的键 scores1 所对应的值的类型是：<class 'list'>
train_loader里的第1个元素的键 image0 所对应的值的类型是：<class 'torch.Tensor'>
train_loader里的第1个元素的键 image1 所对应的值的类型是：<class 'torch.Tensor'>
train_loader里的第1个元素的键 all_matches 所对应的值的类型是：<class 'list'>
train_loader里的第1个元素的键 file_name 所对应的值的类型是：<class 'list'>
----------------------结束监视代码----------------------
```
由此我们就清楚了。`train_loader`是一个可迭代对象。当迭代里面的元素时，得到的是一个字典（上面的输出：`train_loader里的第1个元素的类型： <class 'dict'>`）。这个字典里面每个键和键所对应的值的类型参见上面print出来的内容。关于Dataset和DataLoader的更详细的教程，参见[PyTorch DATASETS & DATALOADERS新手教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)。

下面的一行代码初始化了所要用的模型：
``` python
superglue = SuperGlue(config.get("superglue", {}))
```
我们先来看看，输入到`SuperGlue`类的构造函数中的输入究竟是什么。试运行下述代码：
``` python
print("----------------------开始监视代码----------------------")
print('type(config.get("superglue", {}))：', type(config.get("superglue", {})))
print("----------------------我的分割线1----------------------")
print('config.get("superglue", {})：', config.get("superglue", {}))
print("----------------------结束监视代码----------------------")
exit()
superglue = SuperGlue(config.get("superglue", {}))
```
结果为：
```
----------------------开始监视代码----------------------
type(config.get("superglue", {}))： <class 'dict'>
----------------------我的分割线1----------------------
config.get("superglue", {})： {'weights': 'indoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
----------------------结束监视代码----------------------
```
由此我们就清楚了：要初始化一个`SuperGlue`类的实例，需要提供一个字典。这个字典给出了三个参数：`'weights'`，`'sinkhorn_iterations'`和`'match_threshold'`。我们还学会了一个Python字典的用法。比如，在一个空白脚本里测试一下下述代码：
``` python
mydict = {
    "a": {"aa": 1, "ab": 2, "ac": 3},
    "b": {"ba": 4, "bb": 5, "bc": 6},
}

print(type(mydict.get("b", {})))

print(mydict.get("b", {}))
```
结果为：
```
<class 'dict'>
{'ba': 4, 'bb': 5, 'bc': 6}
```
这样我们就明白了：Python的字典，如果还嵌套了子字典，那可以使用`get`函数来完整地调用子字典（就是上面这段演示代码所演示的用法）。

接下来我们看一看初始化好的`SuperGlue`类的实例`superglue`究竟长什么样子。试运行下述代码：
``` python
superglue = SuperGlue(config.get("superglue", {}))
print("----------------------开始监视代码----------------------")
print("type(superglue)：", type(superglue))
print("----------------------我的分割线1----------------------")
print("superglue：", superglue)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(superglue)： <class 'models.superglue.SuperGlue'>
----------------------我的分割线1----------------------
superglue： SuperGlue(
  (kenc): KeypointEncoder(
    (encoder): Sequential(
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
  )
  (gnn): AttentionalGNN(
    (layers): ModuleList(
      (0): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (1): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (2): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (3): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (4): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (5): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (6): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (7): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (8): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (9): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (10): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (11): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (12): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (13): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (14): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (15): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (16): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (17): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
        )
      )
    )
  )
  (final_proj): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
)
----------------------结束监视代码----------------------
```
至此，我所使用的SuperGlue网络终于展现出了它的完整结构。关于这个网络结构，之后再来详细地研究。现在，我们先继续分析下面的训练代码。