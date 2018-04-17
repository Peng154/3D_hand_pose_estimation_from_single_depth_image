# 3D hand pose estimation from single depth image
- 作者：彭昊 
- 邮箱：826113664@qq.com

## 运行环境
- 系统：Win10/Ubuntu 16.04
- 深度学习库：tensorflow 1.6
- python：3.5
- 其他python库：使用anaconda安装
- 其中openni2在win10下的使用可以参考本文博客教程：[Win10安装OpenNI2并通过python接口调用Kinect](https://blog.csdn.net/peng154/article/details/79127630)

## 使用说明
- 数据集是MSRA15 Hands手势数据集，需要放置在`data/MSRA/`文件夹下，下载链接：[MSRA](https://www.dropbox.com/s/bmx2w0zbnyghtp7/cvpr15_MSRAHandGestureDB.zip?dl=0)
- `src/config.py` 包含网络训练和测试的配置以及各项超参数
- `src/run_testing.py`用于手势识别网络的测试
- `src/run_training.py`用于手势识别网络的训练
- `src/test_runtimepipiline.py`用于调用kinect摄像头实时识别手势

## 两种网络结构
一共有两个手势识别网络结构：CPM_Hands和Res_Encoder_Hands,分别在`src/model/`文件夹的两个py文件中

## 测试集测试结果
- 其中MSRA数据集P1-P8为训练集，P0为测试集
- Res_Encoder_Hands测试及手指关节点误差为7.5mm
- CPM_Hands测试集上手指关节点平均误差仅为6.8mm ！！

## 参考文献
1. [DeepPrior++](https://github.com/moberweger/deep-prior-pp) （本项目部分代码参考此project）
2. [CPM](https://github.com/timctho/convolutional-pose-machines-tensorflow) （本项目部分结构参考此project）

## 最后
- 需要预训练好的模型或者对本项目由疑惑可以联系本人。
- 本项目在未经本人同意情况下，仅用于非商业用途！！！

![Simulation Result](https://raw.githubusercontent.com/Peng154/real-time-path-planning-for-Simbad/master/rt_rrt_star.gif)