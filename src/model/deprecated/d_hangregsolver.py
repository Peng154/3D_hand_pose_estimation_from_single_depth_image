from multiprocessing import Queue

import numpy as np
import tensorflow as tf
import os, cv2

from src.data.dataProducer import DataBatchProducer, DataBatch
from src.model.solver import Solver

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class HandRegSolver(Solver):
    def __init__(self, Xdata, Ydata, params, Cubes, gt3Dcrops, coms, hd, pca=None):
        """

        :param Xdata:
        :param Ydata:
        :param params:
        :param pca: PCA 计算矩阵
        """
        super().__init__(Xdata, Ydata, params)
        self.pca = pca
        self.Cubes = Cubes
        self.data_queue = Queue(maxsize= 50) # 缓冲区大小为50个Batch
        self.gt3Dcrops = gt3Dcrops
        self.coms = coms
        self.hd = hd

        self.batchCubes = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='cube_holder')


    def startLoadData(self):
        """
        开启data agumentation的两个生产者，填充数据缓冲区
        :return:
        """
        if self.params.aug_modes is not None:
            rawData = DataBatch(self.X_data['train'], self.gt3Dcrops['train'], self.Cubes['train'], self.coms['train'])

            indices = np.arange(0, self.X_data['train'].shape[0])
            np.random.shuffle(indices)
            mid = self.X_data['train'].shape[0] // 2

            p1 = DataBatchProducer(indices=indices[:mid], rawData=rawData, batchSize=self.params.batchsize,
                                   hd=self.hd, aug_modes=self.params.aug_modes, queue=self.data_queue)
            p2 = DataBatchProducer(indices=indices[mid:], rawData=rawData, batchSize=self.params.batchsize,
                                   hd=self.hd, aug_modes=self.params.aug_modes, queue=self.data_queue)
            p1.daemon = True
            p1.start()
            print('数据生成进程1开启')
            p2.daemon = True
            p2.start()
            print('数据生成进程2开启')
        else:
            pass

    def loadNextBatch(self, start_idx):
        """
        从缓冲区中获取下一个Batch
        :param idxs:
        :return:
        """
        # idxs = self.train_indicies[start_idx:start_idx + self.params.batchsize]
        # self.batchCubes = self.train_cubes[idxs]
        if self.params.aug_modes is not None:
            # 进程共享空间获取下一batch
            self.dataBatch = self.data_queue.get()
            if self.pca is not None:
                return self.dataBatch.imgs, self.pca.transform(np.reshape(self.dataBatch.gt3Dcrops,
                                                              newshape=[self.dataBatch.gt3Dcrops.shape[0], -1]))

            else:
                return self.dataBatch.imgs, np.reshape(a=self.dataBatch.gt3Dcrops, newshape=(self.dataBatch.gt3Dcrops.shape[0], -1))
        else:
            return super().loadNextBatch(start_idx)

    def getEvaluationOps(self, y_infer, y, type):
        """
        计算手指关节的平均误差
        :param y_infer:
        :param y:
        :return:
        """
        if self.pca is not None:
            # 转回3D
            # 这里还要乘上cube的才是
            pca = tf.constant(self.pca.components_, dtype=tf.float32)
            y_3Dcrop = tf.reshape(tf.matmul(y, pca), [-1, pca.shape[1].value // 3, 3])\
                       * (tf.reshape(self.batchCubes, [-1, 1, 3])/2.)
            y_infer_3Dcrop = tf.reshape(tf.matmul(y_infer, pca), [-1, pca.shape[1].value // 3, 3])\
                             * (tf.reshape(self.batchCubes, [-1, 1, 3])/2.)
        else:
            y_3Dcrop = tf.reshape(y, [-1, y.shape[1].value // 3, 3]) * (tf.reshape(self.batchCubes, [-1, 1, 3])/2.)
            y_infer_3Dcrop = tf.reshape(y_infer, [-1, y_infer.shape[1].value // 3, 3]) * (tf.reshape(self.batchCubes, [-1, 1, 3])/2.)
        # 计算误差
        errors = tf.sqrt(tf.reduce_sum(tf.square(y_3Dcrop - y_infer_3Dcrop), axis=2)) # 每个关节点的误差
        mean_errors = tf.reduce_mean(errors) # 平均误差
        # errors = tf.reduce_mean(tf.square(y - y_infer))
        if type == 'test':
            return [mean_errors, errors]
        else:
            return [mean_errors]

    def getEvalFeedDict(self, d, type='train'):
        feed_dict = dict(d)
        assert self.Cubes[type] is not None, '缺失数据变量：Cubes[{}]'.format(type)
        # print(self.Cubes[type].shape)
        # print(self.idxs.shape)
        if type == 'train' and self.params.aug_modes is not None:
            feed_dict[self.batchCubes] = self.dataBatch.cubes
        else:
            feed_dict[self.batchCubes] = self.Cubes[type][self.idxs]
        return feed_dict

    def printBatchEvaluationResults(self, val_result):
        """

        :param val_result:
        :return:
        """
        error_avg = val_result[0]
        print("and avg_error of {} mm".format(error_avg))

    def printEpochEvaluationResults(self, val_results, type):
        """

        :param val_results:
        :return:
        """
        batch_mean_errors = np.array([val_result[0] for val_result in val_results])
        print("and avg_error of {} mm".format(np.mean(batch_mean_errors)))

        if type != 'test':
            return

        # 画出PCK图像
        batch_joint_errors = np.concatenate([np.array(val_result[1]) for val_result in val_results])
        # print(batch_joint_errors.shape)

        pcks = []
        x = range(81)

        for threshold in x:
            mean_pck = np.mean(batch_joint_errors < threshold)
            pcks.append(mean_pck)

        plt.figure(figsize=(8,8))
        plt.grid(True)
        plt.plot(x, pcks)
        print('------------------------------------')
        plt.savefig('pck.png')

    def saveReuslt(self, x_data, y_infer, resultDir):
        return
        self.count = 1
        batch_size = x_data.shape[0]
        print(x_data.shape)
        print(y_infer.shape)
        if not os.path.exists(resultDir):
            os.makedirs(resultDir, exist_ok=True)

        for i in range(batch_size):
            img = x_data[i] * 255



