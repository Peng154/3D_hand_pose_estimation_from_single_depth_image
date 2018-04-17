import tensorflow as tf
import numpy as np

class LayersHelper(object):
    def __init__(self):
        """

        """
        self.is_training = tf.placeholder(dtype=bool, name='is_training')
        self.iteration = tf.placeholder(dtype=tf.float32, name='iteration')
        self.extra_update_ops = [] # 额外的更新操作，主要用于更新bn层的
        self.weight_decay = tf.placeholder(dtype=tf.float32, name='weight_decay')

    @property
    def get_update_ops(self):
        return self.extra_update_ops

    def getWeight(self, name, shape, stddev=0.1, reg=False, dtype=tf.float32):
        assert isinstance(shape, tuple)
        init = tf.truncated_normal(shape=shape, stddev=stddev)
        if reg:  # 是否使用正则化
            regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
        else:
            regularizer = None
        weight = tf.get_variable(name=name, initializer=init, dtype=dtype, regularizer=regularizer)
        return weight

    def getBias(self, name, shape, dtype=tf.float32):
        assert isinstance(shape, tuple)
        init = tf.constant(0.01, dtype=dtype, shape=shape)
        return tf.get_variable(name=name, initializer=init, dtype=dtype)

    def conv(self, x, filter_num, ksize=(3, 3), stride=1, padding='SAME', reg=False):
        """
        :param x: 卷积输入，shape[N,H,W,C]
        :param filter_num: 整形，卷积核数量，也就是卷积输出维数
        :param ksize: 卷积核大小，例:2*2=(2,2)
        :param stride: 卷积步长
        :param padding: 卷积填充方式，有'SAME'和'VALID'两种
        :return: 卷积后的tensor
        """
        assert isinstance(filter_num, int)
        assert isinstance(ksize, tuple)
        input_dim = x.get_shape()
        filter_dim = ksize + (input_dim[-1].value, filter_num)  # 卷积核维度[H,W,C,N]
        # print(filter_dim)
        fan_in = np.prod(input_dim[1:])
        # print(fan_in)
        # print('????')
        # 这里权重初始化采用某篇论文中大神的测试推荐结果。。。。
        filter = self.getWeight('conv_w', shape=filter_dim, stddev=np.sqrt(2.0 / fan_in.value), reg=reg)
        # filter = self.getWeight('conv_w', shape=filter_dim, stddev=0.1, reg=reg)
        bias = self.getBias('conv_b', shape=(filter_num,))
        conv_out = tf.nn.conv2d(x, filter=filter, strides=(1, stride, stride, 1), padding=padding) + bias

        return conv_out

    def fc(self, x, output_dim, reg=False):
        """
        全连接层
        :param x: 输入数据，维度[N,D]
        :param output_dim: 输出数据单个维度，整形 [O]
        :return: 输出，维度[N, O]
        """
        assert isinstance(output_dim, int)
        input_dim = x.get_shape()
        fan_in = np.prod(input_dim[1:])
        if len(input_dim) > 2:  # 如果超过两维，就reshape
            x = tf.reshape(x, [-1, fan_in.value])
        w = self.getWeight('fc_w', shape=(fan_in.value, output_dim), stddev=np.sqrt(2.0 / fan_in.value), reg=reg)
        # w = self.getWeight('fc_w', shape=(fan_in.value, output_dim), stddev=0.1, reg=reg)
        b = self.getBias('fc_b', shape=(output_dim,))
        return tf.matmul(x, w) + b

    def relu(self, x):
        """
        relu激活函数
        :param x:
        :return:
        """
        return tf.nn.relu(x)

    def conv_relu(self, x, filter_num, ksize=(3, 3), stride=1, padding='SAME', reg=False):
        """
        卷积-relu层
        :param x:
        :param filter_num:
        :param ksize:
        :param stride:
        :param padding:
        :return:
        """
        return self.relu(self.conv(x, filter_num, ksize, stride, padding, reg))

    def max_pool(self, x, ksize=(2, 2), stride=2, padding='SAME'):
        """
        最大值池化层
        :param x:
        :param ksize:
        :param stride:
        :param padding:
        :return:
        """
        filter_dim = (1,) + ksize + (1,)
        return tf.nn.max_pool(x, ksize=filter_dim, strides=(1, stride, stride, 1), padding=padding)

    def bn(self, x):
        """
        batch_normalization 层
        :param x:
        :param is_training:
        :param iteration:
        :return:
        """
        # 计算测试时的滑动平均方差和均值 decay=min(0.999, （1+steps）/（10+steps）)
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, self.iteration)
        gamma = tf.get_variable('bn_gamma', shape=[x.get_shape()[-1]], initializer=tf.ones_initializer)
        beta = tf.get_variable('bn_beta', shape=[x.get_shape()[-1]], initializer=tf.zeros_initializer)

        # tensorboard 可视化
        # tf.summary.histogram(gamma.name, gamma)
        # tf.summary.histogram(beta.name, beta)

        axis = list(range(len(x.get_shape()) - 1))  # 求最低一维的方差
        mean, variance = tf.nn.moments(x, axis)
        update_moving_averages = exp_moving_avg.apply([mean, variance])  # 利用mean和variance来更新滑动参数
        m = tf.cond(self.is_training, lambda: mean,
                    lambda: exp_moving_avg.average(mean))  # average是利用mean为key获取，获取当前shadow value的值
        v = tf.cond(self.is_training, lambda: variance, lambda: exp_moving_avg.average(variance))

        Ybn = tf.nn.batch_normalization(x, m, v, beta, gamma, 1e-5)

        self.extra_update_ops.append(update_moving_averages)

        # tf.summary.histogram('bn_out', Ybn)
        return Ybn

    def drop_out(self, x, keep_prop=0.5):
        """
        dropout 层，防止过拟合
        :param x:
        :param is_training: 是否是训练中，测试中dropout不起作用
        :param keep_prop: cell的保持率
        :return:
        """
        prop = tf.cond(self.is_training, lambda: tf.constant(keep_prop), lambda: tf.constant(1.0))
        y = tf.nn.dropout(x, prop)
        return y

    def res_stack(self, x, block_nums, filters_out, stride, reg=False, bottle_neck=True):
        """

        :param x:
        :param block_nums:
        :param filters_out:
        :param stride: 一个stack的步长，其实也就最开始的卷积层会有步长，其他都为1
        :param bottle_neck: 是否使用“瓶颈结构”
        :return:
        """
        assert block_nums >= 1
        for i in range(block_nums):
            s = stride if i ==0 else 1 # 只有在每个stack开始的第一次卷积步长为输入，其他卷积层步长都为1
            with tf.variable_scope('bolck_{}'.format(i)):
                x = self.res_block(x, filters_out, stride=s, bottle_neck=bottle_neck, reg=reg)
        return x

    def res_block(self, x, filters_out, stride, reg=False, bottle_neck=True):
        """
        一个res_block有两层/三层卷积层
        :param x:
        :param filters_out: 输出的维数，当使用bottle_neck结构的时候，中间conv输出维数为filters_out//4
        :param stride: 步长
        :param bottle_neck:
        :return:
        """
        filters_out_internal = filters_out//4 # bottle_neck 中间卷积层的输出维数
        shortcut = x # 高速连接层
        filters_in = x.get_shape()[-1].value

        if bottle_neck:
            with tf.variable_scope('a'):
                x = self.conv(x, filters_out_internal, ksize=(1, 1), stride=stride, reg=reg)
                x = self.bn(x)
                x = self.relu(x)
            with tf.variable_scope('b'):
                x = self.conv(x, filters_out_internal, ksize=(3,3), stride=1, reg=reg)
                x = self.bn(x)
                x = self.relu(x)
            with tf.variable_scope('c'):
                x = self.conv(x, filters_out, ksize=(1, 1), stride=1, reg=reg)
                x = self.bn(x)
        else:
            with tf.variable_scope('A'):
                x = self.conv(x, filters_out, ksize=(3,3), stride=stride, reg=reg)
                x = self.bn(x)
                x = self.relu(x)
            with tf.variable_scope('B'):
                x = self.conv(x, filters_out, ksize=(3, 3), stride=1, reg=reg)
                x = self.bn(x)

        with tf.variable_scope('shortcut'):
            if filters_in != filters_out or stride != 1:
                shortcut = self.conv(shortcut, filters_out, ksize=(1,1), stride=stride, reg=reg)
                shortcut  =self.bn(shortcut)

        return self.relu(x + shortcut)

    def softmax_loss(self, y_infer, y_g):
        """
        softmax 分类器的损失, 其中y_infer的维度和y_g的维度需要一样
        :param y_infer:
        :param y_g:
        :return:
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_g, logits=y_infer))
        # 正则化损失
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss_ = tf.add_n([loss] + regularization_losses)

        return loss_

    def sqr_loss(self, y_infer, y_g):
        """
        平方差损失，其中y_infer的维度和y_g的维度需要一样
        :param y_infer:
        :param y_g:
        :return:
        """
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_g - y_infer), axis=1))
        # 正则化损失
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss_ = tf.add_n([loss] + regularization_losses)

        return loss_