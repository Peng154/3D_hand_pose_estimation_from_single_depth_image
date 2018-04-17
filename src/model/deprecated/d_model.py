import tensorflow as tf
from model.layers import LayersHelper

class Model(object):
    def __init__(self, cacheFile=None):
        """
        模型类肯定包含输入和输出的placeholder
        """
        self.X = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32)
        self.lh = LayersHelper() # 用于构造模型的层帮助类
        self.cacheFile = cacheFile

    @property
    def get_loss(self):
        return None

    @property
    def get_y_infer(self):
        return None

    def inference(self, X, y):
        """
        构建一个预测模型，
        模型model接受两个参数：
            X：数据数据， y：数据标签
        返回三个参数：
            y_infer：模型预测结果
            loss:模型内部定义的损失
        :param x:
        :param y:
        :return:
        """
        pass