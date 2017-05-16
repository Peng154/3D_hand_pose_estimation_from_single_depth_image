from src.data.importers import ICVLImporter
from src.data.dataset import ICVLDataset
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
import pandas as pd
from src.data_input import TCVLDataInput

class HandPose():
    def __init__(self, dataDir, rng=None, weight_decay=0.001, batchSize=128):
        # 正则化参数
        self.weight_decay = weight_decay
        # 批数据大小
        self.batchSize = batchSize
        if rng is None:
            self.rng = np.random.RandomState(23455)
        else:
            self.rng = rng
        self.dataImporter = ICVLImporter(dataDir)
        # batch 的索引，自动循环
        self._batchIndex = -1

        #########################
        #test
        self.dataInput = TCVLDataInput(dataDir, useCache=True)
        #########################

    def loadData(self):
        self.loadTrainData()
        self.loadTestData()

    def loadTrainData(self):
        # 加载训练数据以及测试数据
        # trainSeq = self.dataImporter.loadSequence('train', ['0'], shuffle=True, rng=self.rng)
        # self.trainSeqs = [trainSeq]
        #
        # # 把训练数据以及测试数据导出成numpy数组
        # trainDataSet = ICVLDataset(self.trainSeqs)
        # # 获取训练数据以及标签，把图片数值归一化到了[-1,1]
        # # val_gt3D 是原始手势关节坐标3D化后，归一化到质心，再归一化到[-1,1]
        # #  数据格式data[数量，维数，height，width];label[数量，关节数目，维度(3)]
        # self._train_data,self._train_gt3D = trainDataSet.imgStackDepthOnly('train')
        # # 转置矩阵[height, width, dimenstion]
        # # print(self._train_data.shape)
        # self._train_data = np.transpose(self._train_data,[0, 2, 3, 1])

        ###################################
        #test
        self._train_data, self._train_gt3D = self.dataInput.loadData(dataName='train', shuffle=True, rng=self.rng)
        self._train_data = np.transpose(self._train_data, [0, 2, 3, 1])
        ###################################

        # 打印图片，查看是否有很多0数据，发现并没有
        temp= self._train_data[0: 128]
        temp = temp.reshape((temp.shape[0], -1))
        temp = pd.DataFrame(temp)
        temp.to_csv('./train.csv')

        size_mb = self._train_data.nbytes /(1024*1024)
        print('load data {}MB'.format(size_mb))

    def loadTestData(self):
        testSeq = self.dataImporter.loadSequence('test_seq_1')
        self.testSeqs = [testSeq]
        testDataSet = ICVLDataset(self.testSeqs)
        self._test_data, self._test_gt3D = testDataSet.imgStackDepthOnly('test_seq_1')
        # print(self._train_data.shape)
        self._test_data = np.transpose(self._test_data, [0, 2, 3, 1])

    def create_embedding(self):
        self.pca = PCA(n_components=30)
        # 转换成shape[数量，48]
        self.pca.fit(self._train_gt3D.reshape((self._train_gt3D.shape[0],self._train_gt3D.shape[1]*3)))
        # 数据降维
        self._train_gt3D_embed = self.pca.transform(
            (self._train_gt3D.reshape(self._train_gt3D.shape[0],self._train_gt3D.shape[1]*3)))
        self._test_gt3D_embed = self.pca.transform(
            self._test_gt3D.reshape(self._test_gt3D.shape[0],self._test_gt3D.shape[1]*3))
        # self._val_gt3D_embed = self.pca.transform(
        #     self._test_gt3D.reshape(self._test_gt3D.shape[0], self._test_gt3D.shape[1] * 3))

    # 创建weights
    def weight_variable(self, shape, wd, stddev):
        # with tf.device("/gpu:0"):
        initial = tf.random_normal(shape=shape, stddev=stddev)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(initial), wd)
            tf.add_to_collection('losses', weight_decay)
        # print(initial.dtype)
        return tf.Variable(initial)

    # 创建bias
    def bias_variable(self, shape):
        # with tf.device("/gpu:0"):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def predict(self, images):
        """
        构建卷积神经网络，输出预测关节
        权值初始化很重要！！！！！
        :param images: 需要预测的图像数据（tensor）
        :return: 返回预测好的关节节点的坐标（tensor）
        """

        # 卷积层1
        with tf.variable_scope('conv1') as scope:
            kernel = self.weight_variable([5, 5, 1, 8], self.weight_decay, stddev=(2./(5*5))**0.5)
            bias = self.bias_variable([8])
            pre_activation = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
            pre_activation = tf.nn.bias_add(pre_activation, bias)
            # batch normalization
            gama = self.bias_variable([1])
            beta = self.bias_variable([1])
            # 计算均值、方差
            mean, variance = tf.nn.moments(pre_activation, axes=[0, 1, 2])
            norm = tf.nn.batch_normalization(pre_activation, mean, variance, beta, gama, 5e-2)
            conv1 = tf.nn.relu(norm, name=scope.name)

        # 池化层1
        pool1 = tf.nn.max_pool(conv1, [1, 4, 4, 1], [1, 4, 4, 1], padding='VALID')

        # 卷积层2
        with tf.variable_scope('conv2') as scope:
            kernel = self.weight_variable([5, 5, 8, 8], self.weight_decay, stddev=(2./(5*5*8))**0.5)
            bias = self.bias_variable([8])
            pre_activation = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
            pre_activation = tf.nn.bias_add(pre_activation, bias)
            # batch normalization
            gama = self.bias_variable([1])
            beta = self.bias_variable([1])
            # 计算均值、方差
            mean, variance = tf.nn.moments(pre_activation, axes=[0, 1, 2])
            norm = tf.nn.batch_normalization(pre_activation, mean, variance, beta, gama, 5e-2)
            conv2 = tf.nn.relu(norm, name=scope.name)

        # 池化层2
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

        # 卷积层3
        with tf.variable_scope('conv3') as scope:
            kernel = self.weight_variable([3, 3, 8, 8], self.weight_decay, stddev=(2./(3*3*8))**0.5)
            bias = self.bias_variable([8])
            pre_activation = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
            pre_activation = tf.nn.bias_add(pre_activation, bias)
            # batch normalization
            gama = self.bias_variable([1])
            beta = self.bias_variable([1])
            # 计算均值、方差
            mean, variance = tf.nn.moments(pre_activation, axes=[0, 1, 2])
            norm = tf.nn.batch_normalization(pre_activation, mean, variance, beta, gama, 5e-2)
            conv3 = tf.nn.relu(norm, name=scope.name)

        # 池化层3,不进行池化
        pool3 = conv3

        # 下面是全链接层
        with tf.variable_scope('local4') as scope:
            # 数据转换成1维
            input = tf.reshape(pool3, [pool3.get_shape()[0].value, -1])
            # 获得单个数据维数
            dim = input.get_shape()[1].value
            weights = self.weight_variable([dim, 1024], self.weight_decay, stddev=(2./(1024))**0.5)
            bias = self.bias_variable([1024])
            pre_activation = tf.nn.bias_add(tf.matmul(input, weights), bias)
            # batch normalization
            gama = self.bias_variable([1])
            beta = self.bias_variable([1])
            # 计算均值、方差
            mean, variance = tf.nn.moments(pre_activation, axes=[0])
            norm = tf.nn.batch_normalization(pre_activation, mean, variance, beta, gama, 5e-2)
            local4 = tf.nn.relu(norm, name=scope.name)

        with tf.variable_scope('local5') as scope:
            weights = self.weight_variable([1024, 1024], self.weight_decay, stddev=(2./(1024))**0.5)
            bias = self.bias_variable([1024])
            pre_activation = tf.nn.bias_add(tf.matmul(local4, weights), bias)
            # batch normalization
            gama = self.bias_variable([1])
            beta = self.bias_variable([1])
            # 计算均值、方差
            mean, variance = tf.nn.moments(pre_activation, axes=[0])
            norm = tf.nn.batch_normalization(pre_activation, mean, variance, beta, gama, 5e-2)
            local5 = tf.nn.relu(norm, name=scope.name)

        with tf.variable_scope('local6') as scope:
            weights = self.weight_variable([1024, 30], self.weight_decay, stddev=(2./(1024))**0.5)
            bias = self.bias_variable([30])
            pre_activation = tf.nn.bias_add(tf.matmul(local5, weights), bias)
            # 最后一层输出层，不需要relu！！！！！神他妈，这个bug我找了一天
            y = pre_activation

        return y

    def loss(self, y, label):
        """
        计算预测得到的关节和标记的损失函数
        :param y: 预测的结果
        :param label: 数据的标签
        :return:
        """
        # 长度一定要相同
        assert y.get_shape()[0] == label.get_shape()[0]
        print(y.get_shape())
        print(label.get_shape())
        # 用差的平方作为损失度量
        cost = tf.reduce_mean(tf.reduce_sum(tf.square(y-label), axis=1))
        # 加上L2正则化项，得到最终的损失方程
        total_loss = tf.add_n(tf.get_collection('losses'))/(2*self.batchSize)
        total_loss += cost
        return total_loss

    def add_Prior(self, y):
        self.pca.components_ = np.cast['float32'](self.pca.components_)
        self.pca.mean_ = np.cast['float32'](self.pca.mean_)
        weights = tf.constant(self.pca.components_)
        bias = tf.constant(self.pca.mean_)
        gt3D = tf.nn.bias_add(tf.matmul(y, weights), bias)
        return gt3D



    # 获取下一个batch
    def getNextBatch(self, type=0):
        '''

        :param type: 请求的类型，0-训练数据；1-测试数据
        训练数据按照batchSize一个一个返回
        测试数据直接返回全部数据
        :return:
        '''
        if type==0:
            data_num = self._train_data.shape[0]
            if(self._batchIndex < data_num//self.batchSize):
                self._batchIndex += 1

                idxs = np.cast['int'](np.random.random_sample(self.batchSize)*self._train_data.shape[0])
                # print(idxs)
                train_data_sample = []
                train_gt3D_sample = []

                for i in idxs:
                    train_data_sample.append(self._train_data[i])
                    train_gt3D_sample.append(self._train_gt3D_embed[i])

                train_data_sample = np.array(train_data_sample)
                train_gt3D_sample = np.array(train_gt3D_sample)

                return train_data_sample,train_gt3D_sample

                # return self._train_data[self._batchIndex*self.batchSize:(self._batchIndex+1)*self.batchSize]\
                #     ,self._train_gt3D_embed[self._batchIndex*self.batchSize: (self._batchIndex+1)*self.batchSize]
            else:
                self.setBatchIndex(-1)
                return None

        elif type == 1:
            data_num = self._test_data.shape[0]
            self.batchSize = data_num
            if (self._batchIndex < data_num // self.batchSize):
                self._batchIndex+=1
                # 打印图片，查看是否相同，发现并不相同
                # temp= self._test_data[self._batchIndex * self.batchSize: (self._batchIndex + 1) * self.batchSize]
                # temp = temp.reshape((temp.shape[0], -1))
                # temp = pd.DataFrame(temp)
                # temp.to_csv('./test.csv')
                return self._test_data[self._batchIndex * self.batchSize: (self._batchIndex + 1) * self.batchSize] \
                    , self._test_gt3D_embed[self._batchIndex * self.batchSize: (self._batchIndex + 1) * self.batchSize]
            else:
                self.setBatchIndex(-1)
                return None
        else:
            raise ValueError('invalid data type')

    def setBatchIndex(self, new_index):
        self._batchIndex = new_index

    @property
    def batch_index(self):
        return self._batchIndex

    @property
    def train_data(self):
        if hasattr(self, '_train_data'):
            return self._train_data
        else:
            raise ValueError('hasn\'t load train data')

    @property
    def teat_data(self):
        if hasattr(self, '_test_data'):
            return self._test_data
        else:
            raise ValueError('hasn\'t load test data')
