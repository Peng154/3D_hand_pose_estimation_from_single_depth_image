# 3D hand pose estimation from single depth image
- ���ߣ���� 
- ���䣺826113664@qq.com

## ���л���
- ϵͳ��Win10/Ubuntu 16.04
- ���ѧϰ�⣺tensorflow 1.6
- python��3.5
- ����python�⣺ʹ��anaconda��װ
- ����openni2��win10�µ�ʹ�ÿ��Բο����Ĳ��ͽ̳̣�[Win10��װOpenNI2��ͨ��python�ӿڵ���Kinect](https://blog.csdn.net/peng154/article/details/79127630)

## ʹ��˵��
- ���ݼ���MSRA15 Hands�������ݼ�����Ҫ������`data/MSRA/`�ļ����£��������ӣ�[MSRA](https://www.dropbox.com/s/bmx2w0zbnyghtp7/cvpr15_MSRAHandGestureDB.zip?dl=0)
- `src/config.py` ��������ѵ���Ͳ��Ե������Լ��������
- `src/run_testing.py`��������ʶ������Ĳ���
- `src/run_training.py`��������ʶ�������ѵ��
- `src/test_runtimepipiline.py`���ڵ���kinect����ͷʵʱʶ������

## ��������ṹ
һ������������ʶ������ṹ��CPM_Hands��Res_Encoder_Hands,�ֱ���`src/model/`�ļ��е�����py�ļ���

## ���Լ����Խ��
- ����MSRA���ݼ�P1-P8Ϊѵ������P0Ϊ���Լ�
- Res_Encoder_Hands���Լ���ָ�ؽڵ����Ϊ7.5mm
- CPM_Hands���Լ�����ָ�ؽڵ�ƽ������Ϊ6.8mm ����

## �ο�����
1. [DeepPrior++](https://github.com/moberweger/deep-prior-pp) ������Ŀ���ִ���ο���project��
2. [CPM](https://github.com/timctho/convolutional-pose-machines-tensorflow) ������Ŀ���ֽṹ�ο���project��

## ���
- ��ҪԤѵ���õ�ģ�ͻ��߶Ա���Ŀ���ɻ������ϵ���ˡ�
- ����Ŀ��δ������ͬ������£������ڷ���ҵ��;������

![Simulation Result](https://raw.githubusercontent.com/Peng154/real-time-path-planning-for-Simbad/master/rt_rrt_star.gif)