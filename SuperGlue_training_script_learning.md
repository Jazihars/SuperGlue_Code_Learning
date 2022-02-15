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
我们可以看到，这两行代码，第一行是初始化一个训练数据集对象`train_set`，第二行是初始化一个训练数据加载器对象`train_loader`。而且，由于这里没有使用[torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)，因此SuperGlue-pytorch的训练会采用单机单卡的方式进行。关于如何实现单机多卡或多机多卡，可以参考一些比较优秀的开源代码（比如[Swin Transformer的代码](https://github.com/Jazihars/Swin_Transformer_Code_Learning/blob/main/Swin_Transformer_Code_Study_Notes.md)）。以后，我会考虑详细地学一下该如何实现单机多卡和多机多卡的编写。（2022年2月15日更新：我已经实现了[单机多卡训练SuperGlue的脚本](https://github.com/Jazihars/SuperGlue_PyTorch_MultiGPU_implementation)）
如果要深入研究关于SuperGlue训练数据的加载，就必须进入到这两行代码里面进行深入的研究。

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

<div id="train_DataLoader_diedai"></div>

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

接下来的代码设置了模型训练的方式为GPU训练：
``` python
if torch.cuda.is_available():
    superglue.cuda()  # make sure it trains on GPU
else:
    print("### CUDA not available ###")
```
由于我的调试是在开发机上进行的，所以显然是在GPU上训练的。经过测试，并没有print出来`### CUDA not available ###`这句话，因此这两行代码其实没有什么好说的。

下面的代码构造了一个优化器和平均loss的列表：
``` python
optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
mean_loss = []
```
优化器是神经网络的训练中必须要使用的。关于PyTorch优化器的教程，可以参看[PYTORCH: OPTIM的一个简单例子](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html)或者[OPTIMIZING MODEL PARAMETERS教程](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)，也可参看[PyTorch优化器文档](https://pytorch.org/docs/stable/optim.html)。在这里，我们只需要知道，初始化一个优化器需要提供哪些参数以及优化器长什么样子就行了。关于优化器的详细用法，之后进入到下面的代码中时会继续深入分析的。
我们先来测试一下下述代码：
``` python
print("----------------------开始监视代码----------------------")
print("type(superglue.parameters())：", type(superglue.parameters()))
print("----------------------我的分割线1----------------------")
print("superglue.parameters()：", superglue.parameters())
print("----------------------结束监视代码----------------------")
exit()
optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
mean_loss = []
```
结果为：
```
----------------------开始监视代码----------------------
type(superglue.parameters())： <class 'generator'>
----------------------我的分割线1----------------------
superglue.parameters()： <generator object Module.parameters at 0x7f632577b820>
----------------------结束监视代码----------------------
```
单纯这样看，我们似乎看不出来什么东西。我们利用vscode的定义跳转功能，进入到这个`superglue.parameters()`函数的里面看一下它的定义：
``` python
# superglue.parameters()函数的定义，位于/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/modules/module.py文件里

def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    r"""Returns an iterator over module parameters.

    This is typically passed to an optimizer.

    Args:
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.

    Yields:
        Parameter: module parameter

    Example::

        >>> for param in model.parameters():
        >>>     print(type(param), param.size())
        <class 'torch.Tensor'> (20L,)
        <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

    """
    for name, param in self.named_parameters(recurse=recurse):
        yield param
```
从这里看，我们就明白了。`superglue.parameters()`函数返回一个模型参数的迭代器，而这个模型参数的迭代器通常会被传递给优化器的构造函数用来初始化优化器（参考[这个教程](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html)）。我们试着迭代一下这个模型参数的迭代器。测试下述代码：
``` python
print("----------------------开始监视代码----------------------")
print("type(superglue.parameters())：", type(superglue.parameters()))
print("----------------------我的分割线1----------------------")
print("superglue.parameters()：", superglue.parameters())
print("----------------------我的分割线2----------------------")
print(
    "isinstance(superglue.parameters(), Generator)：",
    isinstance(superglue.parameters(), Generator),
)
print("----------------------我的分割线3----------------------")
temp_1 = 1
for i in superglue.parameters():
    print(f"superglue.parameters()中的第{temp_1}个参数的类型是：", type(i))
    temp_1 += 1
print("----------------------我的分割线4----------------------")
temp_2 = 1
for j in superglue.parameters():
    print(f"superglue.parameters()中的第{temp_2}个参数是：", j)
    temp_2 += 1
    if temp_2 == 4:
        break
print("----------------------结束监视代码----------------------")
exit()
optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
mean_loss = []
```
结果为：
```
----------------------开始监视代码----------------------
type(superglue.parameters())： <class 'generator'>
----------------------我的分割线1----------------------
superglue.parameters()： <generator object Module.parameters at 0x7f7914951890>
----------------------我的分割线2----------------------
isinstance(superglue.parameters(), Generator)： True
----------------------我的分割线3----------------------
superglue.parameters()中的第1个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第2个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第3个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第4个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第5个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第6个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第7个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第8个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第9个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第10个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第11个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第12个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第13个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第14个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第15个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第16个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第17个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第18个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第19个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第20个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第21个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第22个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第23个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第24个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第25个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第26个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第27个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第28个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第29个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第30个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第31个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第32个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第33个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第34个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第35个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第36个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第37个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第38个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第39个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第40个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第41个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第42个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第43个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第44个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第45个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第46个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第47个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第48个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第49个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第50个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第51个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第52个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第53个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第54个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第55个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第56个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第57个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第58个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第59个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第60个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第61个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第62个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第63个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第64个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第65个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第66个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第67个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第68个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第69个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第70个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第71个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第72个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第73个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第74个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第75个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第76个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第77个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第78个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第79个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第80个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第81个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第82个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第83个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第84个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第85个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第86个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第87个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第88个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第89个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第90个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第91个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第92个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第93个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第94个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第95个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第96个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第97个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第98个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第99个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第100个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第101个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第102个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第103个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第104个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第105个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第106个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第107个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第108个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第109个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第110个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第111个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第112个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第113个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第114个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第115个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第116个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第117个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第118个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第119个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第120个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第121个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第122个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第123个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第124个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第125个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第126个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第127个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第128个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第129个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第130个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第131个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第132个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第133个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第134个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第135个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第136个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第137个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第138个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第139个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第140个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第141个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第142个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第143个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第144个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第145个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第146个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第147个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第148个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第149个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第150个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第151个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第152个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第153个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第154个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第155个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第156个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第157个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第158个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第159个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第160个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第161个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第162个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第163个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第164个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第165个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第166个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第167个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第168个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第169个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第170个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第171个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第172个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第173个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第174个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第175个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第176个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第177个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第178个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第179个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第180个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第181个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第182个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第183个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第184个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第185个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第186个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第187个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第188个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第189个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第190个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第191个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第192个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第193个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第194个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第195个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第196个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第197个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第198个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第199个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第200个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第201个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第202个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第203个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第204个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第205个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第206个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第207个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第208个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第209个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第210个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第211个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第212个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第213个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第214个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第215个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第216个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第217个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第218个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第219个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第220个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第221个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第222个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第223个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第224个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第225个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第226个参数的类型是： <class 'torch.nn.parameter.Parameter'>
superglue.parameters()中的第227个参数的类型是： <class 'torch.nn.parameter.Parameter'>
----------------------我的分割线4----------------------
superglue.parameters()中的第1个参数是： Parameter containing:
tensor(1., device='cuda:0', requires_grad=True)
superglue.parameters()中的第2个参数是： Parameter containing:
tensor([[[-0.1948],
         [-0.3589],
         [-0.0879]],

        [[ 0.4845],
         [ 0.5442],
         [-0.3898]],

        [[-0.0350],
         [-0.4070],
         [-0.3277]],

        [[-0.3535],
         [-0.3371],
         [ 0.5733]],

        [[ 0.5009],
         [-0.2826],
         [-0.5263]],

        [[ 0.0820],
         [ 0.1446],
         [-0.2046]],

        [[ 0.5742],
         [-0.1971],
         [ 0.0546]],

        [[ 0.4568],
         [-0.3885],
         [-0.0119]],

        [[ 0.1558],
         [ 0.1578],
         [ 0.5255]],

        [[ 0.0391],
         [ 0.3348],
         [ 0.0443]],

        [[ 0.5338],
         [ 0.4946],
         [ 0.5081]],

        [[ 0.3786],
         [ 0.2134],
         [ 0.2121]],

        [[ 0.2738],
         [ 0.3530],
         [ 0.0957]],

        [[ 0.1592],
         [-0.0684],
         [-0.2240]],

        [[ 0.1961],
         [-0.5182],
         [-0.4093]],

        [[-0.3641],
         [ 0.4289],
         [ 0.0757]],

        [[-0.2434],
         [ 0.1924],
         [-0.2685]],

        [[ 0.5094],
         [ 0.4531],
         [ 0.1083]],

        [[-0.5599],
         [ 0.2272],
         [ 0.1881]],

        [[-0.1274],
         [-0.0564],
         [ 0.2718]],

        [[ 0.1801],
         [-0.4029],
         [ 0.2975]],

        [[ 0.4354],
         [ 0.3177],
         [-0.3558]],

        [[ 0.5423],
         [-0.0809],
         [ 0.4017]],

        [[-0.0378],
         [-0.0253],
         [ 0.1949]],

        [[ 0.3115],
         [ 0.2656],
         [ 0.0722]],

        [[-0.0941],
         [-0.2266],
         [-0.4042]],

        [[ 0.2118],
         [-0.0269],
         [-0.5650]],

        [[-0.3973],
         [-0.5037],
         [ 0.0301]],

        [[-0.1902],
         [-0.4390],
         [-0.2855]],

        [[ 0.3776],
         [-0.3351],
         [ 0.1666]],

        [[ 0.1841],
         [-0.0898],
         [ 0.3376]],

        [[-0.0060],
         [ 0.2526],
         [-0.1010]]], device='cuda:0', requires_grad=True)
superglue.parameters()中的第3个参数是： Parameter containing:
tensor([-0.4749,  0.1358, -0.2101,  0.5600, -0.3274,  0.2971, -0.3640,  0.4130,
        -0.1853,  0.3061, -0.1347, -0.0546,  0.4334,  0.2403, -0.5243,  0.3647,
        -0.2911,  0.0924,  0.1600, -0.5440, -0.4733,  0.3589, -0.1496,  0.4431,
        -0.1292, -0.2198,  0.2057,  0.1433,  0.3471, -0.4147, -0.2773, -0.4146],
       device='cuda:0', requires_grad=True)
----------------------结束监视代码----------------------
```
至此，我们已经很清楚地看到了传给优化器`torch.optim.Adam`类的构造函数的参数究竟长什么样子。传给优化器`torch.optim.Adam`类的构造函数的神经网络参数对象是一个生成器（`isinstance(superglue.parameters(), Generator)： True`），这个生成器是一个可迭代对象，里面保存了神经网络的各个层的参数。这些参数都是一些PyTorch的张量。**在之后的训练中，就会由这个优化器对象来更新神经网络的参数。**（关于PyTorch优化器的使用，**请务必仔细阅读[这里的教程](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html)**，并把这份教程里的代码自己跑一跑试一试。这份教程写得极为清楚。）

下面我们来看看优化器对象长什么样子。测试如下的代码：
``` python
optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
print("----------------------开始监视代码----------------------")
print("type(optimizer)：", type(optimizer))
print("----------------------我的分割线1----------------------")
print("optimizer：", optimizer)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(optimizer)： <class 'torch.optim.adam.Adam'>
----------------------我的分割线1----------------------
optimizer： Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.0001
    weight_decay: 0
)
----------------------结束监视代码----------------------
```
原来如此，优化器居然是长这个样子的。当把优化器print出来之后，我们可以看到，优化器里面包含了优化器的名字（在我这里是`Adam`）以及优化器包含的各种参数。这些参数的详细含义，我目前还不清楚。如果有需要的话，可能以后会去读读论文，了解一下都有哪些优化器以及他们的作用吧。目前，我暂且先专注在提高代码工程能力上，所以对于优化器的设计细节，我暂且先不深入研究。

下面，我们终于进入了正式训练的部分。下面就是`/SuperGlue-pytorch/train.py`脚本剩下的所有代码了：
``` python
# start training 在这部分训练正式开始
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
首先，在正式训练开始的部分，我们看到了这样的两行代码：
``` python
for epoch in range(1, opt.epoch + 1):
    epoch_loss = 0
    superglue.double().train()
```
这两行代码的第一行是设定了这个epoch的loss。第二行代码的作用是将模型`superglue`的所有浮点参数转换成双精度实型数double数据类型，并把模型调整为训练模式。这里我们要注意：`superglue`模型调用的`double()`和`train()`函数，都是`/mnt/data-2/data/zitong.yin/conda_env/superglue/lib/python3.8/site-packages/torch/nn/modules/module.py`文件的`class Module:`类的函数，意即，这两个函数都是PyTorch官方规定的模型必须具有的函数。参考[PyTorch官方torch.nn.Module类的文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

下面的代码是一个循环遍历了所有训练数据的for循环：
``` python
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
```
注意，通过这份代码的学习，我是要学习训练脚本的编写的，所以要特别留意，这里是怎么循环遍历训练数据的。我们可以看到，循环遍历训练数据是在`for i, pred in enumerate(train_loader):`这句话完成的。这里，对训练用DataLoader进行了枚举，枚举变量是一个指标`i`和一个`pred`。这里涉及到Python自带函数`enumerate()`的用法。我先看看这个函数的用法。在一个空白脚本里，测试如下代码：
``` python
seasons = ["Spring", "Summer", "Fall", "Winter"]

for i, pred in enumerate(seasons, start=1):
    print(f"第{i}个枚举变量是：", pred)
```
结果为：
```
第1个枚举变量是： Spring
第2个枚举变量是： Summer
第3个枚举变量是： Fall
第4个枚举变量是： Winter
```
参考[Python官方文档enumerate()函数的解释](https://docs.python.org/3.8/library/functions.html#enumerate)，就完全清楚了。Python自带的`enumerate()`函数，其实就是对一个可迭代对象进行迭代。只不过，附带上了一个序号而已。所以，上面SuperGlue-pytorch训练脚本里的`for i, pred in enumerate(train_loader):`这句话里的pred，其实就是在对`train_loader`进行遍历迭代。按照[之前对`train_loader`进行遍历迭代的结果](#train_DataLoader_diedai)，可以看到，`train_loader`是一个可迭代对象，当迭代里面的元素时，得到的是一个字典。这个字典里面的键和键所对应的值之前已经print出来了，就是下面的这些：
```
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
```
下面的代码是对`train_loader`中的数据进行一个数据格式的重构。我们上面已经看到了，`train_loader`中很多键所对应的值的类型是`list`。`list`数据类型不适合于后面的操作。所以需要把`list`数据类型转换成`torch.tensor`数据类型。测试下面的代码：
``` python
# start training
for epoch in range(1, opt.epoch + 1):
    epoch_loss = 0
    superglue.double().train()
    for i, pred in enumerate(train_loader):
        temp = 1
        for k in pred.keys():
            if k != "file_name" and k != "image0" and k != "image1":
                if type(pred[k]) == torch.Tensor:
                    pred[k] = pred[k].cuda()
                    print("执行了这里")
                else:
                    print(
                        f"----------------------第{temp}个键开始----------------------"
                    )
                    print("len(pred[k])：", len(pred[k]))
                    print("修改前的pred[k][0].shape：", pred[k][0].shape)
                    print(
                        f"----------------------第{temp}个分割线----------------------"
                    )
                    pred[k] = torch.stack(pred[k]).cuda()
                    print("修改后的pred[k].shape：", pred[k].shape)
                    print(
                        f"----------------------第{temp}个键结束----------------------\n"
                    )
                    temp += 1
        exit()
```
结果为：
```
----------------------第1个键开始----------------------
len(pred[k])： 1
修改前的pred[k][0].shape： torch.Size([1, 389, 2])
----------------------第1个分割线----------------------
修改后的pred[k].shape： torch.Size([1, 1, 389, 2])
----------------------第1个键结束----------------------

----------------------第2个键开始----------------------
len(pred[k])： 1
修改前的pred[k][0].shape： torch.Size([1, 163, 2])
----------------------第2个分割线----------------------
修改后的pred[k].shape： torch.Size([1, 1, 163, 2])
----------------------第2个键结束----------------------

----------------------第3个键开始----------------------
len(pred[k])： 128
修改前的pred[k][0].shape： torch.Size([1, 389])
----------------------第3个分割线----------------------
修改后的pred[k].shape： torch.Size([128, 1, 389])
----------------------第3个键结束----------------------

----------------------第4个键开始----------------------
len(pred[k])： 128
修改前的pred[k][0].shape： torch.Size([1, 163])
----------------------第4个分割线----------------------
修改后的pred[k].shape： torch.Size([128, 1, 163])
----------------------第4个键结束----------------------

----------------------第5个键开始----------------------
len(pred[k])： 389
修改前的pred[k][0].shape： torch.Size([1])
----------------------第5个分割线----------------------
修改后的pred[k].shape： torch.Size([389, 1])
----------------------第5个键结束----------------------

----------------------第6个键开始----------------------
len(pred[k])： 163
修改前的pred[k][0].shape： torch.Size([1])
----------------------第6个分割线----------------------
修改后的pred[k].shape： torch.Size([163, 1])
----------------------第6个键结束----------------------

----------------------第7个键开始----------------------
len(pred[k])： 2
修改前的pred[k][0].shape： torch.Size([1, 485])
----------------------第7个分割线----------------------
修改后的pred[k].shape： torch.Size([2, 1, 485])
----------------------第7个键结束----------------------
```
上面的这段代码是在做这样一件事：对`train_loader`的`file_name`，`image0`和`image1`这三个键所对应的值不做任何修改。对`train_loader`的剩下的键，如果它所对应的值已经是`torch.Tensor`数据类型了，则就不再更改了；如果它所对应的值不是`torch.Tensor`数据类型（则一定是`list`数据类型），则利用`pred[k] = torch.stack(pred[k]).cuda()`这行代码把它所对应的值堆叠后转换成`torch.Tensor`数据类型。这里用到的关键函数`torch.stack`的用法参见[PyTorch TORCH.STACK文档](https://pytorch.org/docs/stable/generated/torch.stack.html)。从上面print出来的结果里也能很显然地看出，`torch.stack`可以把很多**维度相同**的张量在一个新的维度上拼接起来，形成一个新的张量。
为了加深我们对于这里用到的关键函数`torch.stack`的理解，我们在一个空白脚本里进行如下的实验：
``` python
# 以下代码在一个空白脚本里运行
import torch

T1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
T2 = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
mylist = [T1, T2]
jieguo = torch.stack(mylist)

print("len(mylist)：", len(mylist))
print("mylist[0].shape：", mylist[0].shape)
print("jieguo.shape：", jieguo.shape)
print("jieguo：", jieguo)
```
结果为：
```
len(mylist)： 2
mylist[0].shape： torch.Size([3, 3])
jieguo.shape： torch.Size([2, 3, 3])
jieguo： tensor([[[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9]],

        [[10, 20, 30],
         [40, 50, 60],
         [70, 80, 90]]])
```
再把上面用来盛放相同维度的张量的那个`mylist`列表换成元组试试。测试如下的代码：
``` python
# 以下代码在一个空白脚本里运行
import torch

T1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
T2 = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
mytuple = (T1, T2)
jieguo = torch.stack(mytuple)

print("len(mytuple)：", len(mytuple))
print("mytuple[0].shape：", mytuple[0].shape)
print("jieguo.shape：", jieguo.shape)
print("jieguo：", jieguo)
```
结果为：
```
len(mytuple)： 2
mytuple[0].shape： torch.Size([3, 3])
jieguo.shape： torch.Size([2, 3, 3])
jieguo： tensor([[[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9]],

        [[10, 20, 30],
         [40, 50, 60],
         [70, 80, 90]]])
```
我们的实验表明，`torch.stack`函数可以接收一个装有许多相同形状的PyTorch张量的列表或元组作为输入，并把这个列表或元组里的所有相同形状的张量拼接起来，成为一个新的张量。至此，对于PyTorch自带的`torch.stack`函数的用法，我们已经有了一个比较深入全面的理解。

下面的一行代码就是把训练数据输入到模型里面去进行推理的代码：
``` python
data = superglue(pred)
```
我们先来看一下输入模型中的数据。测试如下的代码：
``` python
print("----------------------开始监视代码----------------------")
print("pred的类型是：", type(pred))
for key in pred.keys():
    print(f"字典pred的键 {key} 所对应的值的类型是：", type(pred[key]))
print("----------------------结束监视代码----------------------")
exit()
data = superglue(pred)
```
结果为：
```
----------------------开始监视代码----------------------
pred的类型是： <class 'dict'>
字典pred的键 keypoints0 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 keypoints1 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 descriptors0 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 descriptors1 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 scores0 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 scores1 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 image0 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 image1 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 all_matches 所对应的值的类型是： <class 'torch.Tensor'>
字典pred的键 file_name 所对应的值的类型是： <class 'list'>
----------------------结束监视代码----------------------
```
可以看到，除了最后一个文件名以外，其他的数据都被转换成了`<class 'torch.Tensor'>`类型了。我们看看经过模型推理后的数据长什么样子。测试下述代码：
``` python
data = superglue(pred)
print("----------------------开始监视代码----------------------")
print("data的类型是：", type(data))
print("----------------------我的分割线1----------------------")
for key in data.keys():
    print(f"字典data的键 {key} 所对应的值是：", data[key])
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
data的类型是： <class 'dict'>
----------------------我的分割线1----------------------
字典data的键 matches0 所对应的值是： tensor([-1, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 16, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, 28, -1, -1, -1, -1, -1, -1, 24], device='cuda:0')
字典data的键 matches1 所对应的值是： tensor([  6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1, 335,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 388,  -1,  -1,  -1,
        381,  -1], device='cuda:0')
字典data的键 matching_scores0 所对应的值是： tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5553, 0.0000, 0.0000,
        0.0000, 0.1397, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0942, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0608, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0404, 0.0000, 0.0000, 0.1216, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0367, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0423, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0624, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0402, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.2151, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.1645, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.4336, 0.0000, 0.0000, 0.0000, 0.0000, 0.1178,
        0.0000, 0.4487], device='cuda:0', dtype=torch.float64,
       grad_fn=<SelectBackward0>)
字典data的键 matching_scores1 所对应的值是： tensor([0.5553, 0.0000, 0.1397, 0.0000, 0.0000, 0.0423, 0.0942, 0.0404, 0.0000,
        0.0000, 0.0000, 0.1216, 0.0608, 0.0000, 0.0000, 0.0367, 0.2151, 0.0402,
        0.0000, 0.0624, 0.0000, 0.0000, 0.0000, 0.0000, 0.4487, 0.1178, 0.1645,
        0.0000, 0.4336, 0.0000], device='cuda:0', dtype=torch.float64,
       grad_fn=<SelectBackward0>)
字典data的键 loss 所对应的值是： tensor([0.7299], device='cuda:0', dtype=torch.float64,
       grad_fn=<SelectBackward0>)
字典data的键 skip_train 所对应的值是： False
----------------------结束监视代码----------------------
```
我们可以看到，输入到推理模型中的`pred`就是一个字典；经过模型的推理之后，得到的`data`还是一个字典。`superglue`模型推理后输出的这个`data`字典的具体构成，是由`/SuperGlue-pytorch/models/superglue.py`里面的`class SuperGlue(nn.Module):`类的`def forward(self, data):`函数所规定的。所以，如果要想详细地理解模型的推理过程，就需要深入地研究`/SuperGlue-pytorch/models/superglue.py`里面的`class SuperGlue(nn.Module):`类的`def forward(self, data):`函数的代码。目前，我暂且先不深入地研究，我先专注于训练脚本自身的代码。

经过模型推理之后，下面是这样的三行代码：
``` python
for k, v in pred.items():
    pred[k] = v[0]
pred = {**pred, **data}
```
粗看起来，这三行代码是在对输入到`superglue`模型中的数据`pred`做某种处理。我也不清楚究竟是在做什么处理。我先来测试一下`for`循环究竟遍历了什么。在一个空白脚本中测试下述代码：
``` python
# 以下代码在一个空白脚本里运行
mydict = {"a": 1, "b": 2, "c": 3}

temp = 1
for k, v in mydict.items():
    print(f"第{temp}次循环的k是：", k)
    print(f"第{temp}次循环的v是：", v)
    temp += 1
```
结果为：
```
第1次循环的k是： a
第1次循环的v是： 1
第2次循环的k是： b
第2次循环的v是： 2
第3次循环的k是： c
第3次循环的v是： 3
```
由此我们就明白了，Python字典的`.items()`方法是在用两个变量同时遍历一个字典的键和值。所以，上面的这两行代码：
``` python
for k, v in pred.items():
    pred[k] = v[0]
```
是在丢弃`pred`这个字典的值的多余的维度而只保留每个字典的值的第一维。我们再来看看每个键的值究竟丢弃掉了多少个变量。测试下述代码：
``` python
for k, v in pred.items():
    print(f"字典pred的键 {k} 所对应的值的长度是：", len(v))
    pred[k] = v[0]
exit()
```
结果为：
```
字典pred的键 keypoints0 所对应的值的长度是： 1
字典pred的键 keypoints1 所对应的值的长度是： 1
字典pred的键 descriptors0 所对应的值的长度是： 128
字典pred的键 descriptors1 所对应的值的长度是： 128
字典pred的键 scores0 所对应的值的长度是： 389
字典pred的键 scores1 所对应的值的长度是： 224
字典pred的键 image0 所对应的值的长度是： 1
字典pred的键 image1 所对应的值的长度是： 1
字典pred的键 all_matches 所对应的值的长度是： 2
字典pred的键 file_name 所对应的值的长度是： 1
```
我们可以看到，主要是描述子和得分两项丢弃掉的无用变量比较多。其他的字典pred的键基本上都是保留了之前的样子。下面一行`pred = {**pred, **data}`涉及到Python的双星语法。[参考stackoverflow上对Python单星和双星语法的解释](https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters)，在一个空白脚本中进行下述实验：
``` python
# 以下代码在一个空白脚本里运行
def foo(*args):
    for a in args:
        print(a)


foo(12345)

foo(1, 2, 3, 4, 5)
```
结果为：
```
12345
1
2
3
4
5
```
由此我们明白了，Python中的单星语法可以用来向函数提供多个变量而无需事先确定变量的个数。再来在空白脚本中测试一下双星号的用法：
``` python
# 以下代码在一个空白脚本里运行
def bar(**kwargs):
    for a in kwargs:
        print(a, kwargs[a])


bar(name="one", age=27)
```
结果为：
```
name one
age 27
```
关于Python的这个单星和双星用法，更多的例子参考[stackoverflow上的解释](https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters)。在SuperGlue的训练脚本中，测试如下的代码：
``` python
for k, v in pred.items():
    pred[k] = v[0]
pred = {**pred, **data}
print("----------------------开始监视代码----------------------")
print("type(pred)：", type(pred))
print("----------------------我的分割线1----------------------")
print("pred：", pred)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(pred)： <class 'dict'>
----------------------我的分割线1----------------------
pred： {'keypoints0': tensor([[[  3.8799, 324.7194],
         [  4.0033, 465.9073],
         [  6.4027, 386.8461],
         [  7.0455, 450.2341],
         [  8.6949, 460.7588],
         [  9.3618, 459.4267],
         [  9.9620, 249.7474],
         [ 10.5393, 436.3051],
         [ 13.8779, 476.0050],
         [ 14.7766, 430.1528],
         [ 14.8132, 467.0882],
         [ 16.2006, 402.4373],
         [ 16.2006, 402.4373],
         [ 16.3079, 406.9292],
         [ 16.3853, 440.5720],
         [ 17.9748, 389.1681],
         [ 19.9168, 409.0428],
         [ 20.9812, 457.2615],
         [ 21.7987, 390.2296],
         [ 22.3309, 380.9073],
         [ 24.9365, 387.7922],
         [ 24.9365, 387.7922],
         [ 26.1036, 384.9496],
         [ 26.6571, 412.0014],
         [ 26.6974, 450.6580],
         [ 27.2928, 427.9142],
         [ 29.4721, 429.4547],
         [ 31.6647, 441.7844],
         [ 31.8067, 456.7132],
         [ 31.8067, 456.7132],
         [ 32.7249, 406.5130],
         [ 32.7249, 406.5130],
         [ 34.4353, 468.3237],
         [ 34.5996, 385.5879],
         [ 35.1500, 253.1093],
         [ 36.5360, 430.3878],
         [ 37.7059, 460.6881],
         [ 38.1981, 386.6043],
         [ 39.6826, 417.3599],
         [ 40.1252, 243.5373],
         [ 40.3646, 437.4519],
         [ 41.1582, 441.0442],
         [ 43.8847, 402.6511],
         [ 43.9456, 467.3972],
         [ 43.9648, 441.0357],
         [ 44.1506, 440.9682],
         [ 44.1506, 440.9682],
         [ 44.1883, 459.9358],
         [ 44.1883, 459.9358],
         [ 47.4121, 247.5938],
         [ 49.1599, 434.2876],
         [ 52.4966, 401.4023],
         [ 53.8218, 441.0583],
         [ 54.0811, 445.3618],
         [ 54.7025, 438.1021],
         [ 54.8422, 438.0479],
         [ 54.9387, 406.7494],
         [ 55.2218, 241.3543],
         [ 55.6564, 404.1454],
         [ 55.6564, 404.1454],
         [ 56.1570, 456.7455],
         [ 56.1570, 456.7455],
         [ 59.9155, 396.4140],
         [ 60.0294, 482.0211],
         [ 60.3665, 398.9600],
         [ 61.5487, 450.8306],
         [ 61.5487, 450.8306],
         [ 63.5059, 403.3305],
         [ 64.9190, 395.0590],
         [ 65.6629, 427.3502],
         [ 67.3745, 382.7807],
         [ 68.9852, 447.5858],
         [ 69.7771, 359.8464],
         [ 70.6680, 462.3022],
         [ 72.8771, 239.0218],
         [ 73.3435, 429.4160],
         [ 73.7474, 434.2479],
         [ 73.7474, 434.2479],
         [ 74.9147, 428.5021],
         [ 76.1637, 457.8459],
         [ 77.9625, 422.7383],
         [ 77.9982, 418.4682],
         [ 78.8059, 445.1313],
         [ 79.0125, 452.3533],
         [ 79.0125, 452.3533],
         [ 80.9757, 432.2514],
         [ 82.4882, 423.8049],
         [ 84.0666, 463.8098],
         [ 84.9487, 438.0058],
         [ 86.9356, 389.9835],
         [ 87.8197, 422.0032],
         [ 88.8535, 455.7317],
         [ 90.8095, 442.1112],
         [ 91.8529, 423.5119],
         [ 92.9252, 433.7517],
         [ 95.4100, 257.4298],
         [ 96.2153, 387.6166],
         [ 96.2709, 446.5209],
         [ 96.6425, 242.2238],
         [ 98.1040, 234.6932],
         [101.0843, 248.0796],
         [101.5183, 246.6566],
         [101.5183, 246.6566],
         [101.5233, 424.8056],
         [101.5233, 424.8056],
         [101.9225, 440.2332],
         [102.4028, 401.0038],
         [107.8879, 438.4437],
         [108.5934, 463.7253],
         [109.1521, 435.5313],
         [109.8714, 367.2183],
         [110.1100, 394.6215],
         [110.9060, 444.0399],
         [111.2159, 369.8601],
         [111.7757, 373.3094],
         [113.1671, 414.4352],
         [113.6884, 435.6638],
         [113.6884, 435.6638],
         [116.7577, 419.0059],
         [117.0290, 439.0432],
         [118.3170, 440.0583],
         [118.9631, 412.7390],
         [119.2375, 447.8387],
         [119.4688, 409.0066],
         [121.0858, 441.5368],
         [121.5006, 388.7093],
         [121.6413, 358.1146],
         [122.9322, 479.9011],
         [122.9322, 479.9011],
         [122.9675, 367.9553],
         [123.2394, 365.2294],
         [125.3978, 350.8797],
         [125.5218, 374.3550],
         [125.6532, 456.8707],
         [125.6532, 456.8707],
         [125.9246, 381.2256],
         [126.8150, 426.8879],
         [128.3817, 385.2126],
         [128.3817, 385.2126],
         [128.4684, 391.8542],
         [129.7478, 412.9858],
         [131.0175, 369.2447],
         [132.8632, 382.0603],
         [133.0211, 409.4007],
         [133.1390, 412.7264],
         [134.0628, 469.6810],
         [134.3833, 376.1217],
         [136.6457, 350.9711],
         [138.1514, 421.1082],
         [138.1514, 421.1082],
         [138.6037, 368.0234],
         [138.6037, 368.0234],
         [138.7600, 434.0117],
         [138.7600, 434.0117],
         [139.1745, 363.9078],
         [139.7280, 406.8000],
         [140.0892, 375.8263],
         [140.1487, 463.8819],
         [140.1487, 463.8819],
         [141.4780, 393.0606],
         [142.8883, 342.0544],
         [143.4488, 366.5100],
         [143.8434, 425.4983],
         [143.8434, 425.4983],
         [145.6675, 419.4026],
         [145.8980, 418.9024],
         [146.0096, 438.2342],
         [146.0096, 438.2342],
         [146.5770, 464.3516],
         [147.0441, 410.5293],
         [147.7916, 366.0664],
         [149.2097, 434.2622],
         [149.4193, 430.1969],
         [149.9153, 465.3136],
         [149.9153, 465.3136],
         [150.9732, 442.8648],
         [151.6901, 425.7182],
         [151.8304, 370.3904],
         [153.0073, 365.5092],
         [153.0073, 365.5092],
         [154.3333, 469.7682],
         [156.5306, 406.3453],
         [156.5306, 406.3453],
         [156.6321, 429.1024],
         [156.7044, 364.9679],
         [156.9590, 361.8194],
         [157.3152, 440.1025],
         [157.3152, 440.1025],
         [157.3493, 454.1003],
         [157.5075, 456.4652],
         [160.0717, 374.1875],
         [160.0717, 374.1875],
         [162.1445, 434.1662],
         [162.1445, 434.1662],
         [163.6972, 460.5611],
         [166.9851, 400.1981],
         [167.2988, 466.7043],
         [167.5145, 445.0972],
         [173.1280, 401.2365],
         [173.1280, 401.2365],
         [173.1873, 378.0575],
         [173.9792, 447.8986],
         [175.0260, 407.7638],
         [175.3743, 462.6696],
         [175.3743, 462.6696],
         [178.0312, 414.4759],
         [179.0341, 467.5164],
         [180.7964, 419.7775],
         [181.2102, 460.3563],
         [181.4121, 448.5328],
         [181.4121, 448.5328],
         [182.5014, 454.0699],
         [182.5014, 454.0699],
         [184.9342, 402.8006],
         [185.7792, 458.2975],
         [186.0262, 414.0975],
         [186.0262, 414.0975],
         [186.8210, 474.3892],
         [186.9986, 428.2531],
         [187.1156, 436.1687],
         [187.3232, 447.3069],
         [187.4892, 387.9535],
         [187.4892, 387.9535],
         [187.5627, 430.3588],
         [188.5615, 414.7578],
         [190.7169, 431.8212],
         [192.0525, 456.2596],
         [192.7712, 441.5317],
         [193.6896, 370.8723],
         [194.3070, 356.6501],
         [196.1609, 449.6610],
         [196.1609, 449.6610],
         [196.2778, 427.9281],
         [196.7831, 393.4256],
         [196.8316, 393.5069],
         [197.6200, 420.6208],
         [197.6200, 420.6208],
         [198.0189, 421.8189],
         [198.7803, 387.5337],
         [199.3810, 396.0217],
         [199.5736, 434.1453],
         [200.0390, 463.6265],
         [201.9176, 456.4245],
         [201.9176, 456.4245],
         [202.9080, 367.3668],
         [203.2024, 391.1319],
         [203.5576, 374.4486],
         [206.1005, 384.0305],
         [207.6029, 463.1867],
         [207.9695, 304.6579],
         [209.2445, 436.0413],
         [209.2445, 436.0413],
         [211.9185, 435.6279],
         [212.7295, 311.8194],
         [212.7295, 311.8194],
         [212.8459, 473.7148],
         [217.9489, 474.3670],
         [221.1470, 465.2100],
         [221.1470, 465.2100],
         [221.3062, 426.9225],
         [222.3522, 188.6837],
         [222.5707, 188.9208],
         [223.3923, 195.5491],
         [223.8949, 198.1783],
         [223.9176, 303.1888],
         [224.0161, 207.7396],
         [224.7010, 207.1886],
         [224.9382, 157.4708],
         [225.9247, 457.7733],
         [225.9924, 157.6193],
         [226.0027, 193.8852],
         [226.3164, 183.4893],
         [227.7905, 200.4364],
         [227.7905, 200.4364],
         [227.8255, 463.2507],
         [227.8255, 463.2507],
         [228.9977, 443.4140],
         [229.4091, 434.0977],
         [229.4091, 434.0977],
         [230.2023, 145.1289],
         [230.3271, 202.5858],
         [231.5623, 453.3545],
         [231.7980, 192.3343],
         [232.3557, 203.1374],
         [233.3254, 343.5879],
         [233.6915, 444.8970],
         [234.0233, 430.6329],
         [234.5458, 232.9565],
         [234.9293, 243.5457],
         [234.9834, 210.3825],
         [235.2545, 204.0388],
         [236.8956, 225.3697],
         [236.8956, 225.3697],
         [237.1980, 144.9959],
         [237.4598, 249.7267],
         [238.4404, 241.1181],
         [241.0198, 254.5077],
         [241.8683, 394.3205],
         [242.1713, 241.7633],
         [243.0808, 157.7695],
         [243.5356, 184.9710],
         [243.5525, 239.0080],
         [243.8461, 391.5721],
         [243.9237, 205.7537],
         [243.9237, 205.7537],
         [244.0701, 144.0486],
         [245.9321, 241.6996],
         [245.9321, 241.6996],
         [245.9814, 459.5649],
         [245.9814, 459.5649],
         [246.0172, 226.1733],
         [246.0172, 226.1733],
         [246.0184, 171.9465],
         [249.0911, 243.9672],
         [251.5001, 192.6197],
         [252.0384, 132.2814],
         [253.3225, 181.5763],
         [256.4533, 205.8972],
         [256.5651, 241.5483],
         [257.4034, 235.6220],
         [258.4651, 190.6774],
         [258.5655, 459.2050],
         [258.5655, 459.2050],
         [259.6575, 185.7137],
         [260.1925, 238.9884],
         [260.1925, 238.9884],
         [262.0696, 205.0547],
         [262.0696, 205.0547],
         [263.1985, 198.4735],
         [264.5064, 210.1274],
         [265.1026, 183.9652],
         [270.1475, 453.7896],
         [270.1475, 453.7896],
         [270.8970, 260.5825],
         [271.1006, 160.5804],
         [272.8719, 104.2894],
         [273.3907, 217.6446],
         [274.2108, 464.6701],
         [274.9170, 201.6640],
         [279.0794, 462.9632],
         [283.9403, 114.3568],
         [284.5614, 251.1031],
         [286.9489, 214.4506],
         [288.1005, 196.1010],
         [288.8867, 255.3137],
         [291.3298, 439.4854],
         [291.4013, 378.3448],
         [292.1068, 458.7970],
         [292.3164, 247.8341],
         [292.9269, 262.0131],
         [292.9269, 262.0131],
         [295.3124, 124.8077],
         [295.6026, 249.1312],
         [295.6026, 249.1312],
         [296.1420, 463.5174],
         [296.1420, 463.5174],
         [296.4158, 279.8652],
         [296.4158, 279.8652],
         [296.4357, 386.8578],
         [296.5063, 379.9724],
         [296.6466, 136.2464],
         [297.2918, 322.2462],
         [297.3768, 428.9774],
         [297.4528, 334.0691],
         [297.7314, 263.2544],
         [297.7314, 263.2544],
         [298.5898, 434.5613],
         [298.9975, 468.5334],
         [299.2441, 412.1075],
         [299.2932, 427.6565],
         [299.4872, 133.6567],
         [300.1652, 233.8949],
         [301.3119, 469.9930],
         [301.3119, 469.9930],
         [302.4062, 431.6122],
         [302.7056, 434.5181],
         [307.3746, 427.0945],
         [307.5656, 463.1327],
         [308.2811, 444.2672],
         [312.6056, 439.6245],
         [312.6056, 439.6245],
         [312.9336, 439.3075],
         [313.1274, 449.2143],
         [314.2501, 451.9982],
         [315.6016, 437.5092],
         [321.9744, 438.0107],
         [324.9395, 447.2861],
         [330.0099, 434.8314],
         [330.3356, 462.6057]]], device='cuda:0', dtype=torch.float64), 'keypoints1': tensor([[[  5.2187, 471.8276],
         [  6.5215, 474.9070],
         [  6.5215, 474.9070],
         [ 11.5405, 472.0889],
         [ 15.4053, 479.1000],
         [ 15.4463, 465.8135],
         [ 17.3110, 473.8245],
         [ 19.1620, 483.7752],
         [ 19.2738, 293.5565],
         [ 23.9221, 476.5186],
         [ 23.9221, 476.5186],
         [ 24.3890, 478.2715],
         [ 26.7436, 479.4516],
         [ 26.7436, 479.4516],
         [ 29.7930, 277.2614],
         [ 30.1544, 470.7017],
         [ 32.8207, 471.9265],
         [ 36.9040, 466.4597],
         [ 36.9040, 466.4597],
         [ 41.6571, 278.9282],
         [ 42.5089, 271.2205],
         [ 50.5863, 461.4826],
         [ 50.5863, 461.4826],
         [ 51.1292, 289.4519],
         [ 52.1665, 465.9922],
         [ 55.2312, 429.3805],
         [ 66.7333, 450.8883],
         [ 78.3669, 445.2014],
         [ 82.1861, 445.6001],
         [ 84.8014, 467.6633],
         [ 87.7204, 441.3939],
         [ 88.3238, 479.2443],
         [ 88.3238, 479.2443],
         [ 94.1143, 456.3810],
         [ 94.1143, 456.3810],
         [ 94.4759, 410.1700],
         [ 95.5966, 404.4458],
         [ 98.8758, 433.2312],
         [100.3220, 475.3560],
         [100.3220, 475.3560],
         [100.3220, 475.3560],
         [104.0014, 399.5954],
         [104.1737, 429.1364],
         [104.2417, 455.7672],
         [105.8416, 402.9666],
         [107.7930, 465.5715],
         [115.1860, 417.7464],
         [115.1860, 417.7464],
         [120.7126, 227.4377],
         [122.2326, 230.9294],
         [123.5970, 242.3993],
         [123.5970, 242.3993],
         [125.2492, 413.3649],
         [125.2842, 428.0880],
         [126.4668, 229.4242],
         [126.4668, 229.4242],
         [126.5570, 229.7761],
         [127.8954, 439.5881],
         [132.5951, 416.3254],
         [133.5666, 381.3990],
         [134.6295, 397.4684],
         [134.6295, 397.4684],
         [137.4839, 399.3165],
         [144.1426, 423.0505],
         [144.3814, 393.8285],
         [146.7938, 437.2941],
         [147.3503, 383.6485],
         [148.2234, 404.5721],
         [151.8705, 239.7956],
         [152.4979, 423.4153],
         [153.4988, 341.0845],
         [155.2211, 399.9554],
         [156.8091, 428.4055],
         [157.8349, 386.5928],
         [160.3202, 413.6825],
         [160.9937, 366.7773],
         [164.8943, 260.1868],
         [165.4914, 386.0119],
         [165.9043, 317.1323],
         [167.4831, 359.1593],
         [170.4352, 354.7348],
         [170.8816, 310.3367],
         [171.1504, 381.2637],
         [173.1497, 300.4041],
         [174.9937, 287.6363],
         [175.0799, 294.5695],
         [175.0799, 294.5695],
         [177.3751, 331.2231],
         [177.3751, 331.2231],
         [178.3778, 369.4308],
         [178.8411, 360.3655],
         [180.9820, 374.1685],
         [181.1269, 306.8417],
         [181.1269, 306.8417],
         [181.7062, 365.3388],
         [181.7062, 365.3388],
         [181.8531, 389.9884],
         [181.8531, 389.9884],
         [182.1794, 340.5110],
         [182.5325, 310.1730],
         [182.5325, 310.1730],
         [182.9481, 318.6253],
         [183.2457, 278.9173],
         [184.3916, 297.8059],
         [184.3916, 297.8059],
         [186.0428, 363.8514],
         [186.1317, 310.5516],
         [187.7674, 316.3378],
         [187.7674, 316.3378],
         [188.1716, 383.1054],
         [188.5181, 321.9592],
         [189.3745, 271.9246],
         [190.0350, 298.7939],
         [190.2493, 305.1158],
         [190.2493, 305.1158],
         [190.2955, 367.8911],
         [190.2955, 367.8911],
         [192.5970, 289.3468],
         [192.5970, 289.3468],
         [193.5510, 351.6133],
         [193.8373, 374.5072],
         [194.4449, 357.7176],
         [196.0013, 324.0447],
         [196.8256, 313.3221],
         [197.4997, 381.0184],
         [197.8354, 368.1919],
         [197.8354, 368.1919],
         [199.2437, 396.9096],
         [200.3868, 281.7105],
         [206.0812, 281.4382],
         [206.0812, 281.4382],
         [206.7708, 325.5317],
         [208.4344, 387.4865],
         [209.8497, 333.3385],
         [209.8497, 333.3385],
         [213.5500, 327.8336],
         [214.5110, 333.2300],
         [216.2999, 325.2068],
         [216.4937, 379.5533],
         [216.7505, 319.3665],
         [218.9707, 359.3672],
         [219.1615, 348.5063],
         [219.2906, 333.1298],
         [220.4738, 352.1975],
         [220.9525, 304.4603],
         [221.1135, 317.8037],
         [221.1135, 317.8037],
         [221.5667, 286.2429],
         [221.9030, 122.4658],
         [222.9872, 371.7278],
         [223.4228, 347.8241],
         [223.8422, 115.0294],
         [227.0551, 135.7677],
         [227.0781, 327.2112],
         [228.0102, 143.8070],
         [228.4136, 141.1633],
         [228.9094, 347.5731],
         [228.9650, 107.8041],
         [229.0032, 286.0326],
         [229.0032, 286.0326],
         [230.2718, 141.9850],
         [230.3074, 148.1912],
         [231.7320, 119.4420],
         [231.7320, 119.4420],
         [231.9973, 288.8356],
         [232.2840, 138.6487],
         [233.8954,  67.6648],
         [233.9155, 326.5443],
         [233.9594, 144.2698],
         [234.4291, 319.6520],
         [235.2165, 357.1849],
         [235.5253, 154.7715],
         [237.3486, 248.5949],
         [237.5036, 308.7379],
         [237.7169, 131.9591],
         [237.7633, 268.3551],
         [240.1824, 301.2935],
         [240.2339, 345.5522],
         [240.3773, 337.5813],
         [240.5385, 346.5965],
         [240.5616, 158.8872],
         [240.6376, 154.1096],
         [240.7285, 313.6755],
         [240.7285, 313.6755],
         [240.7672, 285.8889],
         [240.7672, 285.8889],
         [241.6067, 285.6976],
         [242.4448,  48.2168],
         [243.0951, 145.8083],
         [243.1634, 249.8996],
         [243.1893, 306.0477],
         [243.3116, 323.9268],
         [243.8293, 194.0837],
         [243.9975, 147.6365],
         [243.9975, 147.6365],
         [244.0058, 264.9874],
         [244.1126, 243.1327],
         [244.5663, 161.8065],
         [244.8534, 293.8186],
         [245.3661, 166.9711],
         [245.3677, 322.6240],
         [245.3677, 322.6240],
         [245.9257, 199.9550],
         [246.5170, 290.4621],
         [247.0147, 317.5222],
         [247.0147, 317.5222],
         [247.0401, 198.9317],
         [247.2119, 284.0141],
         [247.7850, 303.7654],
         [248.0086, 131.0181],
         [248.1187, 128.3215],
         [248.8573, 114.6647],
         [248.8600,  89.4964],
         [248.9879, 293.2399],
         [249.2200, 314.3650],
         [250.5742, 313.2429],
         [251.0479, 298.2709],
         [252.0624, 133.6818],
         [252.1848, 137.1335],
         [252.8680, 307.8665],
         [254.3047, 139.3742],
         [254.5807, 164.7747],
         [255.0214, 156.1723],
         [255.0214, 156.1723],
         [257.3964,  84.0816],
         [260.0229,  45.4396],
         [260.0229,  45.4396],
         [260.1190, 300.3797],
         [260.2067, 138.4947],
         [261.5706, 282.6901],
         [263.9087, 130.6177],
         [264.0681, 134.2312],
         [265.3475, 292.3119],
         [267.0994, 113.8963],
         [267.7112, 126.7934],
         [272.9713, 241.8356],
         [273.5739, 158.0671],
         [274.4726, 275.3428],
         [274.5759, 147.0458],
         [274.5952, 153.5866],
         [274.8827, 297.6512],
         [275.2617, 264.9387],
         [275.3956, 284.4814],
         [276.1856, 159.8094],
         [277.3969, 286.8831],
         [278.3549, 145.7409],
         [281.9001, 162.9595],
         [281.9001, 162.9595],
         [286.1842, 174.2274],
         [289.0045, 181.9618],
         [290.8989, 187.4190],
         [292.1256, 189.4429],
         [296.2355, 204.0939],
         [298.2323, 257.9221],
         [298.2323, 257.9221],
         [302.4943, 257.4051],
         [303.8552, 237.6692],
         [303.9665, 226.6009],
         [307.5126, 233.8709],
         [307.5126, 233.8709],
         [309.2155, 239.7500],
         [310.2464, 226.9147],
         [311.5081, 248.9499],
         [313.2149, 234.2250],
         [313.9880, 261.2924],
         [315.9063, 243.3369],
         [318.2893, 230.7978],
         [318.2893, 230.7978],
         [322.0600, 251.7785],
         [323.7293, 233.2283],
         [325.7702, 236.1737]]], device='cuda:0', dtype=torch.float64), 'descriptors0': tensor([[0.0000, 0.0195, 0.0117, 0.0000, 0.0078, 0.0078, 0.2617, 0.0117, 0.0156,
         0.0000, 0.1289, 0.0625, 0.0156, 0.0312, 0.2656, 0.1016, 0.1719, 0.0000,
         0.0156, 0.1016, 0.0000, 0.3281, 0.2070, 0.0000, 0.4922, 0.3359, 0.0156,
         0.0000, 0.0039, 0.0039, 0.0000, 0.4844, 0.0156, 0.3672, 0.0625, 0.0547,
         0.0039, 0.2617, 0.1328, 0.0000, 0.0039, 0.0039, 0.1172, 0.0430, 0.1016,
         0.0000, 0.5156, 0.0000, 0.2656, 0.1602, 0.0977, 0.1602, 0.0234, 0.0234,
         0.0039, 0.0039, 0.0469, 0.0000, 0.0000, 0.0117, 0.0078, 0.0000, 0.0781,
         0.2617, 0.3359, 0.0000, 0.0703, 0.0039, 0.1484, 0.0078, 0.0938, 0.0117,
         0.0000, 0.0195, 0.3008, 0.0820, 0.0703, 0.0156, 0.0703, 0.0234, 0.0117,
         0.1094, 0.3281, 0.3242, 0.3984, 0.0820, 0.0352, 0.0039, 0.1055, 0.3008,
         0.0703, 0.0508, 0.0000, 0.0703, 0.5000, 0.0000, 0.3242, 0.0078, 0.1055,
         0.5156, 0.0000, 0.0000, 0.0078, 0.3945, 0.1289, 0.0117, 0.0000, 0.1797,
         0.0781, 0.0586, 0.5703, 0.3125, 0.0273, 0.2266, 0.0547, 0.0195, 0.0312,
         0.0117, 0.0391, 0.0039, 0.0000, 0.0000, 0.0156, 0.1523, 0.1328, 0.0664,
         0.0000, 0.0625, 0.1953, 0.0195, 0.2578, 0.0312, 0.1445, 0.0664, 0.0195,
         0.2852, 0.0039, 0.0273, 0.0859, 0.0156, 0.0117, 0.1484, 0.0977, 0.0430,
         0.0000, 0.0078, 0.4922, 0.1484, 0.0000, 0.0078, 0.0000, 0.0039, 0.0352,
         0.0195, 0.0195, 0.1758, 0.4023, 0.0312, 0.0547, 0.0039, 0.1172, 0.0000,
         0.0000, 0.1680, 0.0625, 0.0977, 0.0000, 0.0000, 0.0000, 0.0508, 0.0000,
         0.0156, 0.0742, 0.0195, 0.0273, 0.1211, 0.2227, 0.0000, 0.0000, 0.0000,
         0.0234, 0.1250, 0.0430, 0.0000, 0.0195, 0.3008, 0.2031, 0.0000, 0.1016,
         0.0312, 0.0039, 0.0039, 0.0195, 0.0273, 0.0000, 0.0117, 0.5078, 0.0195,
         0.4648, 0.4883, 0.1680, 0.1680, 0.0156, 0.0039, 0.1562, 0.0156, 0.0234,
         0.0156, 0.0430, 0.0000, 0.3867, 0.0000, 0.0000, 0.0820, 0.0234, 0.0000,
         0.0273, 0.0039, 0.4922, 0.4922, 0.2578, 0.2500, 0.0352, 0.0312, 0.0039,
         0.0078, 0.0000, 0.4141, 0.0078, 0.0117, 0.3516, 0.0781, 0.0898, 0.0156,
         0.0117, 0.0352, 0.0000, 0.0625, 0.0000, 0.0352, 0.0352, 0.0000, 0.1445,
         0.0117, 0.3789, 0.2383, 0.2266, 0.1953, 0.0195, 0.1055, 0.0117, 0.0234,
         0.0234, 0.0039, 0.0000, 0.0234, 0.0625, 0.0000, 0.0117, 0.5000, 0.4844,
         0.5391, 0.0508, 0.3750, 0.5430, 0.0039, 0.0117, 0.0391, 0.0000, 0.2188,
         0.0039, 0.0000, 0.0000, 0.1211, 0.1211, 0.3867, 0.4805, 0.0000, 0.0156,
         0.0078, 0.0469, 0.1367, 0.0000, 0.1250, 0.1172, 0.0898, 0.0000, 0.0039,
         0.0078, 0.0039, 0.0039, 0.3438, 0.0039, 0.3477, 0.0000, 0.0039, 0.0000,
         0.0000, 0.0117, 0.0000, 0.0430, 0.0117, 0.0039, 0.0508, 0.0664, 0.0430,
         0.0312, 0.0000, 0.3086, 0.0273, 0.0039, 0.2930, 0.0664, 0.0000, 0.0117,
         0.2969, 0.0039, 0.0117, 0.0000, 0.1016, 0.0234, 0.0117, 0.4141, 0.0195,
         0.0000, 0.0000, 0.4688, 0.0000, 0.1172, 0.0000, 0.0430, 0.2656, 0.0820,
         0.0391, 0.0000, 0.0078, 0.0000, 0.0078, 0.0391, 0.0938, 0.1719, 0.0352,
         0.0352, 0.0156, 0.0039, 0.0000, 0.4492, 0.0117, 0.0078, 0.0000, 0.0859,
         0.3008, 0.0000, 0.0000, 0.0000, 0.0547, 0.0000, 0.0000, 0.0000, 0.0039,
         0.0000, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000, 0.0156, 0.0039, 0.0000,
         0.0312, 0.0039, 0.0430, 0.0000, 0.0234, 0.0156, 0.0078, 0.2930, 0.1250,
         0.0039, 0.0000, 0.1211, 0.1875, 0.0000, 0.0469, 0.0000, 0.0977, 0.0000,
         0.0352, 0.1680]], device='cuda:0'), 'descriptors1': tensor([[0.0000, 0.1758, 0.0234, 0.2344, 0.1406, 0.2266, 0.0195, 0.0000, 0.0898,
         0.1562, 0.0078, 0.0312, 0.0352, 0.0234, 0.0000, 0.2969, 0.0312, 0.0000,
         0.4648, 0.1484, 0.4336, 0.0000, 0.0312, 0.0312, 0.0195, 0.0039, 0.0430,
         0.0312, 0.2109, 0.0000, 0.1602, 0.0000, 0.0000, 0.0625, 0.0312, 0.2773,
         0.2734, 0.2539, 0.0469, 0.0000, 0.0586, 0.0156, 0.2695, 0.0000, 0.0586,
         0.0039, 0.0312, 0.0000, 0.1016, 0.0000, 0.0352, 0.0000, 0.0742, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0039, 0.0820, 0.1562, 0.1172, 0.0000, 0.0430,
         0.0469, 0.1406, 0.4844, 0.0781, 0.2617, 0.0117, 0.0039, 0.2617, 0.0234,
         0.0000, 0.3320, 0.0000, 0.0195, 0.1133, 0.0000, 0.0430, 0.0195, 0.0469,
         0.0898, 0.1953, 0.2773, 0.0039, 0.0039, 0.4570, 0.2891, 0.0039, 0.0039,
         0.0039, 0.0078, 0.0000, 0.0312, 0.1719, 0.0000, 0.3867, 0.1719, 0.0312,
         0.0195, 0.0469, 0.0859, 0.1250, 0.1133, 0.0117, 0.0000, 0.0742, 0.0820,
         0.0352, 0.0039, 0.0352, 0.0547, 0.0234, 0.0352, 0.1719, 0.1797, 0.0000,
         0.0000, 0.0078, 0.0000, 0.3359, 0.0000, 0.0078, 0.0117, 0.0117, 0.0703,
         0.0039, 0.1289, 0.0000, 0.0000, 0.0117, 0.0000, 0.2539, 0.0078, 0.0117,
         0.0000, 0.0039, 0.0117, 0.1914, 0.0039, 0.2188, 0.0000, 0.1523, 0.0195,
         0.0586, 0.0000, 0.0039, 0.0234, 0.0195, 0.1914, 0.0234, 0.0078, 0.0000,
         0.1680, 0.3008, 0.0273, 0.0820, 0.0195, 0.2266, 0.4297, 0.0977, 0.0156,
         0.0000, 0.0039, 0.1094, 0.0039, 0.0000, 0.1094, 0.0703, 0.1992, 0.1055,
         0.0039, 0.0000, 0.2148, 0.0586, 0.1797, 0.0039, 0.1016, 0.1055, 0.0664,
         0.0000, 0.2422, 0.0117, 0.0039, 0.0000, 0.0000, 0.0000, 0.0781, 0.0117,
         0.0273, 0.0000, 0.0195, 0.1914, 0.0039, 0.5547, 0.2852, 0.0273, 0.0000,
         0.5625, 0.0000, 0.1016, 0.0000, 0.5547, 0.5469, 0.0078, 0.0234, 0.1797,
         0.0000, 0.4922, 0.0664, 0.0234, 0.0000, 0.1367, 0.5195, 0.0078, 0.1914,
         0.5156, 0.1289, 0.0000, 0.0000, 0.0078, 0.0430, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0820, 0.3867, 0.0391, 0.3047, 0.0000, 0.1250, 0.5117, 0.1211,
         0.0039, 0.0156, 0.0000, 0.4531, 0.0000, 0.0195, 0.1484, 0.0000, 0.0039,
         0.0469, 0.3945, 0.1328, 0.4688, 0.0273, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.4766, 0.2383, 0.4805, 0.0195, 0.1562, 0.0117, 0.0508, 0.0742,
         0.2266, 0.2461, 0.0156, 0.1445, 0.1211, 0.1016, 0.1602, 0.1953, 0.0742,
         0.0000]], device='cuda:0'), 'scores0': tensor([0.0271], device='cuda:0', dtype=torch.float64), 'scores1': tensor([0.0150], device='cuda:0', dtype=torch.float64), 'image0': tensor([[[0.7922, 0.7922, 0.7922,  ..., 0.7490, 0.7569, 0.7608],
         [0.7961, 0.7961, 0.7961,  ..., 0.7373, 0.7451, 0.7490],
         [0.8039, 0.8039, 0.8039,  ..., 0.7333, 0.7412, 0.7451],
         ...,
         [0.6588, 0.6549, 0.6549,  ..., 0.7373, 0.7451, 0.7529],
         [0.6824, 0.6745, 0.6745,  ..., 0.7373, 0.7451, 0.7529],
         [0.6980, 0.6902, 0.6824,  ..., 0.7373, 0.7451, 0.7529]]],
       device='cuda:0', dtype=torch.float64), 'image1': tensor([[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         ...,
         [0.2588, 0.2510, 0.2549,  ..., 0.0000, 0.0000, 0.0000],
         [0.2627, 0.2627, 0.2745,  ..., 0.0000, 0.0000, 0.0000],
         [0.2745, 0.2745, 0.2824,  ..., 0.0000, 0.0000, 0.0000]]],
       device='cuda:0', dtype=torch.float64), 'all_matches': tensor([[ 11,  39,  33,  45,  58,  64,  53,  98, 100,  80,  83,  90,  87,  94,
         114, 105, 131, 126, 109, 112, 132, 116, 108, 121, 135, 141, 119, 137,
         139, 146, 142, 122, 150, 143, 133, 170, 177, 162, 166, 172, 176, 175,
         168, 183, 267, 173, 279, 271, 263, 270, 180, 198, 272, 266, 299, 202,
         282, 283, 197, 300, 221, 287, 291, 201, 215, 246, 203, 233, 244, 295,
         223, 294, 206, 264, 208, 220, 320, 323, 334, 340, 225, 214, 227, 328,
         326, 226, 329, 318, 242, 336, 250, 297, 276, 348, 344, 256, 286, 268,
         274, 356, 361, 363, 331, 339, 345, 366, 376, 354, 378, 377, 386, 388,
           0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  13,  14,
          15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,
          29,  30,  31,  32,  34,  35,  36,  37,  38,  40,  41,  42,  43,  44,
          46,  47,  48,  49,  50,  51,  52,  54,  55,  56,  57,  59,  60,  61,
          62,  63,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
          77,  78,  79,  81,  82,  84,  85,  86,  88,  89,  91,  92,  93,  95,
          96,  97,  99, 101, 102, 103, 104, 106, 107, 110, 111, 113, 115, 117,
         118, 120, 123, 124, 125, 127, 128, 129, 130, 134, 136, 138, 140, 144,
         145, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
         161, 163, 164, 165, 167, 169, 171, 174, 178, 179, 181, 182, 184, 185,
         186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 199, 200, 204,
         205, 207, 209, 210, 211, 212, 213, 216, 217, 218, 219, 222, 224, 228,
         229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 243, 245,
         247, 248, 249, 251, 252, 253, 254, 255, 257, 258, 259, 260, 261, 262,
         265, 269, 273, 275, 277, 278, 280, 281, 284, 285, 288, 289, 290, 292,
         293, 296, 298, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
         312, 313, 314, 315, 316, 317, 319, 321, 322, 324, 325, 327, 330, 332,
         333, 335, 337, 338, 341, 342, 343, 346, 347, 349, 350, 351, 352, 353,
         355, 357, 358, 359, 360, 362, 364, 365, 367, 368, 369, 370, 371, 372,
         373, 374, 375, 379, 380, 381, 382, 383, 384, 385, 387, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389,
         389, 389]], device='cuda:0'), 'file_name': '/mnt/data-2/data/zitong.yin/coco2014/train2014/COCO_train2014_000000346550.jpg', 'matches0': tensor([  8,  -1,  23,  -1,  -1,   0,  20,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  14,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1, 187,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1, 166,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  79,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 226,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 249,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 257,  -1,  -1,
         -1, 251,  -1,  -1,  -1,  -1,  -1, 252,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1],
       device='cuda:0'), 'matches1': tensor([  5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   0,  -1,  -1,  -1,  -1,  -1,
         39,  -1,  -1,  -1,  -1,  -1,   6,  -1,  -1,   2,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 127,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  99,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  74,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1, 279,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 348,  -1, 365,
        371,  -1,  -1,  -1,  -1, 361,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
         -1,  -1,  -1,  -1,  -1], device='cuda:0'), 'matching_scores0': tensor([0.4241, 0.0000, 0.2860, 0.0000, 0.1002, 0.2478, 0.3040, 0.1187, 0.0000,
        0.0000, 0.1356, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.1460, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0514, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1539, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.3991, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.1307, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0312, 0.0000, 0.0000, 0.0939,
        0.0000, 0.0000, 0.4779, 0.0000, 0.0000, 0.0000, 0.0861, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0579, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0376, 0.0000, 0.1347,
        0.6246, 0.0000, 0.0000, 0.1333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0627, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0713, 0.0000, 0.0702,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0856, 0.0000, 0.0000, 0.0000,
        0.0667, 0.2151, 0.1656, 0.0000, 0.0000, 0.0949, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0470, 0.0000, 0.0331,
        0.0000, 0.0000, 0.0000, 0.1993, 0.0000, 0.0000, 0.0661, 0.0000, 0.0000,
        0.0706, 0.0754, 0.0436, 0.0000, 0.0000, 0.1405, 0.0000, 0.0000, 0.0000,
        0.0000, 0.1061, 0.0000, 0.0000, 0.0000, 0.0000, 0.0552, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0360, 0.0516, 0.0767, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0352, 0.0000, 0.0000, 0.0332, 0.0882, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0526, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0602, 0.0456, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1595,
        0.0000, 0.1181, 0.0000, 0.0000, 0.1733, 0.0488, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0328, 0.0441, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0373, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0224, 0.0376, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0478, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0664, 0.0000, 0.1513, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0422, 0.0000, 0.1538, 0.0860, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.2071, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0600, 0.0000, 0.0000, 0.1973, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.1540, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0652, 0.0000, 0.0000, 0.0309, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.1980, 0.0000, 0.0000, 0.1102, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.1259, 0.0000, 0.0000, 0.0633, 0.0000, 0.0000,
        0.1204, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0683, 0.0916,
        0.0845, 0.1020, 0.0000, 0.0000, 0.0979, 0.0000, 0.2390, 0.1473, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0964, 0.2012, 0.0000, 0.0000, 0.1193, 0.2329, 0.0000, 0.1637, 0.0000,
        0.0000, 0.0000, 0.2456, 0.1572, 0.0000, 0.0000, 0.0000, 0.0326, 0.1018,
        0.0000, 0.0000, 0.0000, 0.0746, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000], device='cuda:0', dtype=torch.float64,
       grad_fn=<SelectBackward0>), 'matching_scores1': tensor([0.2478, 0.0000, 0.0000, 0.1187, 0.1356, 0.0000, 0.0000, 0.0000, 0.4241,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3991, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.3040, 0.0000, 0.1002, 0.2860, 0.0000, 0.0514, 0.0000,
        0.0579, 0.0000, 0.0000, 0.0000, 0.0000, 0.1181, 0.0000, 0.0652, 0.1307,
        0.0312, 0.0000, 0.0000, 0.0000, 0.0000, 0.0939, 0.0000, 0.0000, 0.0000,
        0.0552, 0.0702, 0.0627, 0.0000, 0.1460, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.1539, 0.0000, 0.0000, 0.1656, 0.0000, 0.1061, 0.0856,
        0.1405, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0376, 0.0000,
        0.1595, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0713, 0.2151, 0.0000,
        0.0000, 0.0000, 0.0754, 0.0949, 0.0667, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0516, 0.0706, 0.0000, 0.0000, 0.0882, 0.0000, 0.0000, 0.0000, 0.0352,
        0.0000, 0.0000, 0.0436, 0.1993, 0.0000, 0.0470, 0.0360, 0.0331, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0526, 0.0456, 0.0767, 0.0000, 0.1733,
        0.0661, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0332, 0.0488, 0.0000,
        0.0000, 0.1513, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0376, 0.0602, 0.0224, 0.0664, 0.0328, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0373, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1347, 0.0000,
        0.0000, 0.0000, 0.0422, 0.0309, 0.0000, 0.0478, 0.0000, 0.0000, 0.0000,
        0.1538, 0.1333, 0.0441, 0.0000, 0.6246, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.1473, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0861, 0.4779, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.1540, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0600, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1980,
        0.1973, 0.2071, 0.0000, 0.0000, 0.0000, 0.0683, 0.0860, 0.0000, 0.0000,
        0.0964, 0.0916, 0.0000, 0.0000, 0.1259, 0.0000, 0.0746, 0.0000, 0.0000,
        0.0633, 0.0326, 0.1020, 0.0000, 0.1102, 0.0845, 0.2390, 0.0000, 0.2329,
        0.2456, 0.0000, 0.0000, 0.0979, 0.0000, 0.2012, 0.0000, 0.0000, 0.0000,
        0.1204, 0.0000, 0.1193, 0.1572, 0.1018, 0.0000, 0.0000, 0.0000, 0.0000,
        0.1637], device='cuda:0', dtype=torch.float64,
       grad_fn=<SelectBackward0>), 'loss': tensor([3.1253], device='cuda:0', dtype=torch.float64,
       grad_fn=<SelectBackward0>), 'skip_train': False}
----------------------结束监视代码----------------------
```
可以看到，经过`pred = {**pred, **data}`这行代码，`pred`字典就变成原先的`pred`和`data`的合并了。所以，这句话其实就是对`pred`字典的格式做了一个整合，把两个字典整合成一个字典。在空白脚本中测试一下下述代码：
``` python
# 以下代码在一个空白脚本里运行
mydict1 = {"a": 1, "b": 2, "c": 3}
mydict2 = {"d": 4, "e": 5, "f": 6}

mydict = {**mydict1, **mydict2}

print(mydict)
```
结果为：
```
{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
```
由此可见，当双星`**`用于Python的字典前面的时候，可以把字典里面的元素提取出来，合并成一个新字典。
**（补充说明一句：以后对于开源代码里我不熟悉的Python或PyTorch用法，我都要尽量自己设计一些例子来复现，增强自己对不熟悉用法的理解。）**

下面的两行代码做了一个是否有匹配特征点的判定：
``` python
if pred["skip_train"] == True:  # image has no keypoint
    continue
```
这两行本身没有什么难度。但是，有一个问题必须搞清楚：在哪里对`pred["skip_train"]`的值做了改动？在`/SuperGlue-pytorch/train.py`脚本里，没有看到对`pred["skip_train"]`的值做改动的部分。这个时候，可以用vscode自带的全局搜索功能，在整个`SuperGlue-pytorch`代码库文件夹里搜索`skip_train`，可以看到，在`/SuperGlue-pytorch/models/superglue.py`文件的`class SuperGlue(nn.Module):`类的`def forward(self, data):`函数里，设定了`'skip_train': True`或`'skip_train': False`。至此就清楚了，对`pred["skip_train"]`的值做改动的部分位于模型的推理代码（也就是`/SuperGlue-pytorch/models/superglue.py`文件的`class SuperGlue(nn.Module):`类的`def forward(self, data):`函数）里面。

下面就是loss计算和梯度反向传播的代码：
``` python
# process loss
Loss = pred["loss"]
epoch_loss += Loss.item()
mean_loss.append(Loss)

# superglue.zero_grad()  #注意，这里我怀疑原始的开源代码作者写错了。我改成了optimizer.zero_grad()
optimizer.zero_grad()
Loss.backward()
optimizer.step()
```
这六行代码整体来说，属于PyTorch自身的固定用法，并没有什么特别的技巧可言，只需要记住会用就行了（参考[这个例子](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html)）。不过，我们还是要来仔细地看一下`Loss`变量究竟是什么样子的。测试下述代码：
``` python
# process loss
Loss = pred["loss"]
print("----------------------开始监视代码----------------------")
print("type(Loss)：", type(Loss))
print("----------------------我的分割线1----------------------")
print("Loss：", Loss)
print("----------------------我的分割线2----------------------")
print("type(Loss.item())：", type(Loss.item()))
print("----------------------我的分割线3----------------------")
print("Loss.item()：", Loss.item())
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(Loss)： <class 'torch.Tensor'>
----------------------我的分割线1----------------------
Loss： tensor([1.9400], device='cuda:0', dtype=torch.float64,
       grad_fn=<SelectBackward0>)
----------------------我的分割线2----------------------
type(Loss.item())： <class 'float'>
----------------------我的分割线3----------------------
Loss.item()： 1.940043657850143
----------------------结束监视代码----------------------
```
可以看到，这里的`Loss`变量不是一个实数，而是一个PyTorch张量。这个PyTorch张量里面记录了loss的值、所在的设备（这里是`cuda`）、数据类型（这里是`torch.float64`）和梯度函数（这里是`<SelectBackward0>`）。而使用函数`.item()`，就可以把`Loss`里的值取出来了。我们在空白脚本里测试一下函数`.item()`的用法：
``` python
# 以下代码在一个空白脚本里运行
import torch

mytensor = torch.tensor([1.9876])

print("mytensor：", mytensor)
print("mytensor.item()：", mytensor.item())
```
结果为：
```
mytensor： tensor([1.9876])
mytensor.item()： 1.9875999689102173
```
（注：在这个演示里，浮点数的位数变多了，这个和计算机自身浮点数存储有关。参考[这里的解释](https://zhuanlan.zhihu.com/p/345487165)。要想知道更详细的解释，可以参考计算机科学的教材，去学学计算机是如何存储浮点数的。）


接下来的代码可视化了训练过程和训练loss：
``` python
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
```
这段代码整体而言，也没什么可说的。`torch.stack`的用法之前已经分析过了。`torch.mean`的用法参考[PyTorch官方TORCH.MEAN文档](https://pytorch.org/docs/stable/generated/torch.mean.html)。关于`torch.mean`的用法，我们在空白脚本里测试一下下述代码：
``` python
# 以下代码在一个空白脚本里运行
import torch

mytensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
mytensor2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
mytensor3 = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])


print("----------------------开始监视代码----------------------")
print("mytensor1：", mytensor1)
print("mytensor1.shape：", mytensor1.shape)
print("torch.mean(mytensor1)：", torch.mean(mytensor1))
print("----------------------我的分割线1----------------------")
print("mytensor2：", mytensor2)
print("mytensor2.shape：", mytensor2.shape)
print("torch.mean(mytensor2)：", torch.mean(mytensor2))
print("----------------------我的分割线2----------------------")
print("mytensor3：", mytensor3)
print("mytensor3.shape：", mytensor3.shape)
print("torch.mean(mytensor3)：", torch.mean(mytensor3))
print("----------------------结束监视代码----------------------")
```
结果为：
```
----------------------开始监视代码----------------------
mytensor1： tensor([1., 2., 3., 4., 5., 6.])
mytensor1.shape： torch.Size([6])
torch.mean(mytensor1)： tensor(3.5000)
----------------------我的分割线1----------------------
mytensor2： tensor([[1., 2., 3.],
        [4., 5., 6.]])
mytensor2.shape： torch.Size([2, 3])
torch.mean(mytensor2)： tensor(3.5000)
----------------------我的分割线2----------------------
mytensor3： tensor([[1.],
        [2.],
        [3.],
        [4.],
        [5.],
        [6.]])
mytensor3.shape： torch.Size([6, 1])
torch.mean(mytensor3)： tensor(3.5000)
----------------------结束监视代码----------------------
```
由此就明白了：`torch.mean()`函数的用法是返回输入张量的所有元素的平均值，而不在乎输入张量的形状如何。

接下来的代码进入了评测阶段：
``` python
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
```
这些代码是在把原来的GPU模式转换成CPU模式，并且把PyTorch张量转换成numpy数组。我们来看看，这些代码里引用的那些函数都有什么功能。测试如下的代码：
``` python
print("----------------------开始监视代码----------------------")
print('type(pred["image0"])：', type(pred["image0"]))
print("----------------------我的分割线1----------------------")
print('pred["image0"].shape：', pred["image0"].shape)
print("----------------------我的分割线2----------------------")
print(
    'type(pred["image0"].cpu().numpy())：',
    type(pred["image0"].cpu().numpy()),
)
print("----------------------我的分割线3----------------------")
print(
    'pred["image0"].cpu().numpy().shape：',
    pred["image0"].cpu().numpy().shape,
)
print("----------------------结束监视代码----------------------")
exit()

image0, image1 = (
    pred["image0"].cpu().numpy()[0] * 255.0,
    pred["image1"].cpu().numpy()[0] * 255.0,
)
```
结果为：
```
----------------------开始监视代码----------------------
type(pred["image0"])： <class 'torch.Tensor'>
----------------------我的分割线1----------------------
pred["image0"].shape： torch.Size([1, 428, 640])
----------------------我的分割线2----------------------
type(pred["image0"].cpu().numpy())： <class 'numpy.ndarray'>
----------------------我的分割线3----------------------
pred["image0"].cpu().numpy().shape： (1, 428, 640)
----------------------结束监视代码----------------------
```
我们可以看到，对一个PyTorch张量调用了`.cpu().numpy()`函数之后，这个张量的形状保持不变，但是数据类型就被转换成了`<class 'numpy.ndarray'>`类型了。

再来测试一下下面的`.detach()`函数的用法。测试如下的代码：
``` python
print("----------------------开始监视代码----------------------")
print('type(pred["matches0"])：', type(pred["matches0"]))
print("----------------------我的分割线1----------------------")
print('pred["matches0"].shape：', pred["matches0"].shape)
print("----------------------我的分割线2----------------------")
print(
    'type(pred["matches0"].cpu().detach().numpy())：',
    type(pred["matches0"].cpu().detach().numpy()),
)
print("----------------------我的分割线3----------------------")
print(
    'pred["matches0"].cpu().detach().numpy().shape：',
    pred["matches0"].cpu().detach().numpy().shape,
)
print("----------------------结束监视代码----------------------")
exit()

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
```
结果为：
```
----------------------开始监视代码----------------------
type(pred["matches0"])： <class 'torch.Tensor'>
----------------------我的分割线1----------------------
pred["matches0"].shape： torch.Size([1024])
----------------------我的分割线2----------------------
type(pred["matches0"].cpu().detach().numpy())： <class 'numpy.ndarray'>
----------------------我的分割线3----------------------
pred["matches0"].cpu().detach().numpy().shape： (1024,)
----------------------结束监视代码----------------------
```
这里我们没有见过的就是PyTorch张量通过点语法调用`detach()`函数的用法。关于这个PyTorch张量的`detach()`函数的用法，可以参考[PyTorch官方TORCH.TENSOR.DETACH文档](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html)。我们在一个空白脚本里测试一下这个函数。在空白脚本里测试如下的代码：
``` python
# 以下代码在一个空白脚本里运行
import torch

mytensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

shuchu = mytensor.cpu().detach().numpy()

print("type(shuchu)：", type(shuchu))
print("shuchu：", shuchu)
```
结果为：
```
type(shuchu)： <class 'numpy.ndarray'>
shuchu： [[1. 2. 3.]
 [4. 5. 6.]]
```
这里的测试，确实看不出来`torch.tensor.detach()`函数的作用，但是参考[上面的文档](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html)可以知道，`torch.tensor.detach()`函数的作用是：返回一个新的张量，这个新的张量是与当前的计算图解绑的。

接下来的两行代码调用了一个新的函数：
``` python
image0 = read_image_modified(image0, opt.resize, opt.resize_float)
image1 = read_image_modified(image1, opt.resize, opt.resize_float)
```
我们来看一下这个`read_image_modified()`函数的完整代码。这个`read_image_modified()`函数位于`/SuperGlue-pytorch/models/utils.py`代码里。`read_image_modified()`函数的完整代码如下：
``` python
def read_image_modified(image, resize, resize_float):
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')
    return image
```
这个函数的具体用法，我暂时先不深入研究了。只需注意：这个函数接收的两个参数`resize`和`resize_float`的含义，在`/SuperGlue-pytorch/train.py`脚本最上面已经有了相应的解释，因此我即便是不完全清楚这个`read_image_modified(image, resize, resize_float)`函数的用法，也可以参照`/SuperGlue-pytorch/train.py`脚本最上面对`resize`和`resize_float`这两个参数的解释来操作。

接下来的几行代码是比较`numpy`数组中的元素与-1的大小，并取出合法的匹配对。如果大于-1，则算是成功的特征点匹配：
``` python
valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]
```
对于这里第一行`valid = matches > -1`的用法，我们在空白脚本里进行如下的测试：
``` python
# 以下代码在一个空白脚本里运行
import numpy as np
import torch

myabc = np.array([1, 2, 3, 4])
mydef = myabc > 0
print("myabc：", myabc)
print("mydef：", mydef)

mytensor1 = torch.tensor([1, 2, 3, 4])
mytensor2 = mytensor1 > 0
print("mytensor1：", mytensor1)
print("mytensor2：", mytensor2)
```
结果为：
```
myabc： [1 2 3 4]
mydef： [ True  True  True  True]
mytensor1： tensor([1, 2, 3, 4])
mytensor2： tensor([True, True, True, True])
```
这种把`numpy`n维数组和PyTorch张量和一个固定数进行逐个元素比较大小的用法，一定要掌握。

剩下的代码，我看了之后，发现没有什么特别重要的需要现在立刻掌握的用法了。我的这份SuperGlue训练脚本学习笔记，就暂且写到这里吧。