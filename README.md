# Introduction





本项目是基于YOLOv3的5G嵌入式智能分类垃圾桶系统。该垃圾桶具有两个功能选项，可以通过连接本地USB摄像头获取图像，也可以通过5G网络获取网络摄像头的图像。需要预先获取5G摄像头的IP地址以及端口。



![](/picture/2.png)

以上为垃圾桶的简易检测界面。

# Requirements

| 操作系统 | Ubuntu 18.04 LTS                            |
| -------- | ------------------------------------------- |
| CPU      | Intel i7-6700U，8GB运行内存，可加速至3.5GHZ |
| GPU      | GTX 1080Ti，12GB显存                        |
| 开发平台 | pycharm + keil5 (python3 + c51)             |
| 开发板   | STC89C52RC                                  |

关于python依赖库安装使用如下命令：

```python
pip3 install -r requirements.txt
```

如需移植到自己的电脑，需要修改cap.py文件里面的相关路径名，以及USB串口的路径。

需预先将单片机代码烧录进51单片机中，如何通过串口连接服务器。

![](/picture/1.jpg)

如图为整个系统的框架图，需要用到两个数码舵机和一个工业摄像头，以及一个5V的稳压模块，和一个电源模块，以及一个单片机最小模块。

# Training

**Start Training：**`python3 train.py`  to begin training after downloading the dataset.

**Resume Training**:`python3 train.py --resume` to resume training from `weights/xxx.pt`

![](/picture/chart.png)

如图为训练loss损失曲线。

# Test



# 



```
pip3 install -r requirements.txt
```



# Training

##### Star Traning:

```python
python3 train.py
```

![](/picture/chart.png)





![](/pictute/1.jpg)



# 








