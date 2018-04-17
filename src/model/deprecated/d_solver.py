import tensorflow as tf
import os
import numpy as np


class SolverParams(object):
    def __init__(self, epochs=1, batchsize=64, printEvery=100, weight_decay=None, optimizer='AdamOptimizer', opt_params={}, aug_modes=None):
        """

        :param epochs: 训练周期
        :param batchsize: 批处理大小
        :param printEvery: 打印代数间隔
        :param opt_params: 优化器参数字典：learning_rate：学习速率，decay: 学习速率衰减， momentum：更新动量
        :param optimizer: 优化器类型：目前有'AdamOptimizer'和'RMSPropOptimizer'
        """
        self.epochs = epochs
        self.batchsize = batchsize
        self.optimizer = optimizer
        self.opt_params = opt_params
        self.printEvery = printEvery
        self.weight_decay = weight_decay
        self.aug_modes = aug_modes

class Solver(object):
    def __init__(self, Xdata, Ydata, params):
        """

        :param Xdata:输入的X数据，是一个字典，包含key：{'train','val','test'} ,对应训练集、验证集、测试集
        :param Ydata: 输入的y数据，是一个字典，包含key：{'train','val','test'} ,对应训练集、验证集、测试集
        :param params:求解超参数
        """
        self.X_data = Xdata
        self.y_data = Ydata
        self.params = params

    def startLoadData(self):
        """
        开始加载数据，可以在这里开启数据的生产者进程
        或者进行一些其他加载数据前需要进行的操作
        :return:
        """
        pass

    def loadNextBatch(self, start_idx):
        """
        加载下一个miniBatch
        :return: batch_X, batch_y
        """
        self.idxs = self.train_indicies[start_idx:start_idx + self.params.batchsize]
        return self.X_data['train'][self.idxs], self.y_data['train'][self.idxs]

    def getEvaluationOps(self, y_infer, y, type):
        correct_prediction = tf.equal(tf.argmax(y_infer, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return [accuracy]

    def printBatchEvaluationResults(self, val_result):
        """
        输出一个miniBatch的评估结果函数
        :param val_result:
        :return:
        """
        accuracy = val_result[0]
        print("and accuracy of {}".format(accuracy))

    def printEpochEvaluationResults(self, val_results, type):
        """
        输出整个周期的评估结果函数
        :param val_results:
        :return:
        """
        accuracies = np.array(val_results)
        print("and accuracy of {}".format(np.mean(accuracies)))


    def saveModel(self, saver, sess, model_path):
        saver.save(sess, model_path)

    def loadModel(self, saver, sess, model_path):
        saver.restore(sess, model_path)

    def getEvalFeedDict(self, dict, type='train'):
        return dict


    def train(self, model, draw=True, cacheModel=True, cacheDir='../train_cache'):
        """
        模型model接受两个参数：
            X：数据数据， y：数据标签
        返回三个参数：
            y_infer：模型预测结果
            loss:模型内部定义的损失
            lh：用于构造模型的LayersHelper
        :param model: 自定义的模型
        :param draw: 是否使用tansorboard画图
        :param cacheModel:是否使用缓存模型
        :param cacheDir: 缓存保存文件夹
        :return:
        """
        X = model.X
        y = model.y
        lr_holder = tf.placeholder(dtype=tf.float32, name='lr_holder')
        lh = model.lh
        y_infer, loss = model.inference(X, y)
        tf.summary.scalar('loss', loss)

        is_training = lh.is_training
        iteration = lh.iteration
        weight_decay = lh.weight_decay

        os.makedirs(cacheDir, exist_ok=True)
        model_path = '{}/model.cpkt'.format(cacheDir)

        # 训练单元
        assert hasattr(tf.train, self.params.optimizer), '未定义的优化方法{}'.format(self.params.optimizer)
        Optimizer = getattr(tf.train, self.params.optimizer)
        optimizer = Optimizer(**self.params.opt_params)
        # optimizer = tf.train.RMSPropOptimizer(lr_holder, decay=self.params.lr_decay, momentum=self.params.momentum)
        train_step = optimizer.minimize(loss)

        print("start train")
        training_now = True
        iter_count = 0
        # have tensorflow compute accuracy
        eval_ops = self.getEvaluationOps(y_infer, y, type='train')

        # shuffle indicies
        self.train_indicies = np.arange(self.X_data['train'].shape[0])
        np.random.shuffle(self.train_indicies)

        # 训练速率
        lr = self.params.opt_params['learning_rate']
        # 运行节点
        ops = [loss, train_step]

        print('开始加载数据...')
        self.startLoadData()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs', sess.graph)

        # 可能加载缓存模型
        saver = tf.train.Saver()
        if cacheModel:
            if os.path.exists('{}.index'.format(model_path)):
                print('加载模型。。。。')
                self.loadModel(saver,sess,model_path)


        assert self.X_data['train'] is not None and self.y_data['train'] is not None
        # 训练集上进行训练
        for e in range(self.params.epochs):
            evaluation_results = [] # 一个周期之内，每个batch的训练测试结果
            losses = [] # 一个周期之内，每个batch的训练损失
            for i in range(int(np.ceil(self.X_data['train'].shape[0]/self.params.batchsize))):
                start_idx = (i * self.params.batchsize) % self.X_data['train'].shape[0]
                # print(self.X_data['train'][idx].shape)
                X_, y_ = self.loadNextBatch(start_idx)
                feed_dict = {
                    X:X_,
                    y:y_,
                    lr_holder:lr,
                    is_training:training_now,
                    iteration:iter_count
                }
                if self.params.weight_decay is not None:
                    feed_dict[weight_decay]=self.params.weight_decay

                if len(lh.extra_update_ops) != 0:
                    sess.run(lh.extra_update_ops, feed_dict=feed_dict)

                if draw:
                    result = sess.run(merged, feed_dict=feed_dict)  # 计算需要写入的日志数据
                    if iter_count % 50 == 0:
                        writer.add_summary(result, i)  # 将日志数据写入文件

                eval_dict = self.getEvalFeedDict(feed_dict)
                # if e == 1:
                #     for k, v in eval_dict.items():
                #         print(k, ' -------', v)
                eval_result = sess.run(eval_ops, feed_dict=eval_dict) # 运行训练评估参数
                evaluation_results.append(eval_result)
                l, _ = sess.run(ops, feed_dict=feed_dict) # 损失以及BP
                losses.append(l)


                if training_now and iter_count%self.params.printEvery ==0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g}" \
                          .format(iter_count, l))
                    self.printBatchEvaluationResults(eval_result) # 输出单个batch的训练评估结果
                iter_count += 1

            # total_correct = np.mean(accuracies)
            total_loss = np.mean(losses)

            # 学习速率衰减
            # lr *= self.params.lr_decay

            print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss, e + 1))
            self.printEpochEvaluationResults(evaluation_results, type='train') # 输出整个epoch的训练评估结果
            # 如果有验证数据集
            if self.X_data['val'] is not None:
                self.test(sess, model, type='val')

        # 如果有验证数据集
        if self.X_data['val'] is not None:
            self.test(sess,  model, type='val')

        # 如果有测试数据集
        if self.X_data['test'] is not None:
            self.test(sess,  model, type='test')

        if cacheModel:
            print('保存模型。。。。')
            self.saveModel(saver, sess, model_path)

    def saveReuslt(self, x_data, y_infer, resultDir):
        """
        将结果保存到指定路径
        :param x_data:
        :param eval_result:
        :param resultDir:
        :return:
        """
        pass

    def test(self, sess, model, type='test', load_model=False, resultDir=None):
        """

        :param sess:
        :param model:
        :param type:
        :param load_model:
        :param resultDir: 将测试结果和手势图片叠加后可视化图片的存储路径
        :return:
        """
        if type == 'test':
            xd = self.X_data['test']
            yd = self.y_data['test']
        elif type == 'val':
            xd = self.X_data['val']
            yd = self.y_data['val']
        else:
            raise NotImplementedError

        lh = model.lh
        y_infer = model.get_y_infer()
        loss = model.get_loss()
        X = model.X
        y = model.y
        training_now = False
        is_training = lh.is_training
        weight_decay = lh.weight_decay

        # print(y_infer)
        # print(y)
        if load_model:
            # 代表第一次加载模型，还没有计算过
            y_infer, loss = model.inference(X, y)

        eval_ops = self.getEvaluationOps(y_infer, y, type)
        ops = [loss, y_infer]

        idxs = np.arange(xd.shape[0])

        evaluation_results = []  # 一个周期之内，每个batch的训练测试结果
        losses = []  # 一个周期之内，每个batch的训练损失

        if load_model:
            assert model.cacheFile is not None
            saver = tf.train.Saver()
            self.loadModel(saver, sess, model.cacheFile)

        for i in range(int(np.ceil(xd.shape[0] / self.params.batchsize))):
            start_idx = (i * self.params.batchsize) % xd.shape[0]
            self.idxs = idxs[start_idx:start_idx + self.params.batchsize]
            # print(self.X_data['train'][idx].shape)
            feed_dict = {
                X: xd[self.idxs],
                y: yd[self.idxs],
                is_training: training_now,
            }
            if self.params.weight_decay is not None:
                feed_dict[weight_decay] = self.params.weight_decay

            eval_dict = self.getEvalFeedDict(feed_dict, type=type)
            eval_result = sess.run(eval_ops, feed_dict=eval_dict)  # 运行训练评估参数
            # print(len(eval_result))
            evaluation_results.append(eval_result)
            ops_result = sess.run(ops, feed_dict=feed_dict)
            losses.append(ops_result[0])
            if resultDir is not None:
                self.saveReuslt(xd[self.idxs], ops_result[1], resultDir)# 保存结果

        total_loss = np.mean(losses)

        print("In {1} : Overall loss = {0:.3g}".format(total_loss, type))
        self.printEpochEvaluationResults(evaluation_results, type)