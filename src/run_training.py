import numpy as np
import gc, os, time
import importlib
import tensorflow as tf
from src.config import FLAGS
from data.importers import MSRA15Importer
from data.dataset import MSRA15Dataset

# 引入自定义的模型模块
model_module = importlib.import_module('src.model.'+ FLAGS.network_def)

# from src.model.cpm_hand_model import CPM_Model

class Data_Generator(object):
    def __init__(self, batch_size, images, labels, cubes):
        """
        数据生成类，可用于训练时随机抽取一个batch的数据
        :param batch_size: batch大小
        :param images: 原始图像数据
        :param labels: 原始标签数据
        :param cubes:  原始框数据
        """
        self.batch_size = batch_size
        self.images = images
        self.labels = labels
        self.cubes = cubes
        self.data_size = images.shape[0]

    def next(self):
        """
        获取下一个batch
        :return:
        """
        index = np.random.choice(a=self.data_size, size=self.batch_size)
        batch_images = self.images[index]
        batch_labels = self.labels[index]
        batch_cubes = self.cubes[index]
        return batch_images,batch_labels,batch_cubes

def main(argv):

    rng = np.random.RandomState(23455)

    print("create data")
    aug_modes = ['com', 'rot', 'none']  # 'sc',

    comref = None  # "./eval/MSRA15_COM_AUGMENT/net_MSRA15_COM_AUGMENT.pkl"
    docom = False
    di = MSRA15Importer('../data/MSRA15/', refineNet=comref)
    seqs = []
    seqs.append(di.loadSequence('P0', shuffle=True, rng=rng, docom=docom))
    seqs.append(di.loadSequence('P1', shuffle=True, rng=rng, docom=docom))
    seqs.append(di.loadSequence('P2', shuffle=True, rng=rng, docom=docom))
    seqs.append(di.loadSequence('P3', shuffle=True, rng=rng, docom=docom))
    seqs.append(di.loadSequence('P4', shuffle=True, rng=rng, docom=docom))
    seqs.append(di.loadSequence('P5', shuffle=True, rng=rng, docom=docom))
    seqs.append(di.loadSequence('P6', shuffle=True, rng=rng, docom=docom))
    seqs.append(di.loadSequence('P7', shuffle=True, rng=rng, docom=docom))
    seqs.append(di.loadSequence('P8', shuffle=True, rng=rng, docom=docom))

    testSeqs = [seqs[0]]
    trainSeqs = [seq for seq in seqs if seq not in testSeqs]

    print("training: {}".format(' '.join([s.name for s in trainSeqs])))
    print("testing: {}".format(' '.join([s.name for s in testSeqs])))

    # create training data
    trainDataSet = MSRA15Dataset(trainSeqs, localCache=False)
    nSamp = np.sum([len(s.data) for s in trainSeqs])
    d1, g1 = trainDataSet.imgStackDepthOnly(trainSeqs[0].name)  # 在这里进行归一化处理
    # 存储所有数据（[-1,1]）
    train_data = np.ones((nSamp, d1.shape[1], d1.shape[2], d1.shape[3]), dtype='float32')
    # 存储所有的标签([-1,1])
    train_gt3D = np.ones((nSamp, g1.shape[1], g1.shape[2]), dtype='float32')
    # 存储所有的立体限制框
    train_data_cube = np.ones((nSamp, 3), dtype='float32')
    # 存储所有的中心点
    train_data_com = np.ones((nSamp, 3), dtype='float32')
    # 存储所有的3D点
    train_gt3Dcrop = np.ones_like(train_gt3D)
    del d1, g1
    gc.collect()
    gc.collect()
    gc.collect()
    oldIdx = 0
    for seq in trainSeqs:
        d, g = trainDataSet.imgStackDepthOnly(seq.name)
        train_data[oldIdx:oldIdx + d.shape[0]] = d
        train_gt3D[oldIdx:oldIdx + d.shape[0]] = g
        train_data_cube[oldIdx:oldIdx + d.shape[0]] = np.asarray([seq.config['cube']] * d.shape[0])
        train_data_com[oldIdx:oldIdx + d.shape[0]] = np.asarray([da.com for da in seq.data])
        train_gt3Dcrop[oldIdx:oldIdx + d.shape[0]] = np.asarray([da.gt3Dcrop for da in seq.data])
        oldIdx += d.shape[0]
        del d, g
        gc.collect()
        gc.collect()
        gc.collect()

    mb = (train_data.nbytes) / (1024 * 1024)
    print("data size: {}Mb".format(mb))

    testDataSet = MSRA15Dataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly(testSeqs[0].name)
    test_data_cube = np.asarray([testSeqs[0].config['cube']] * test_data.shape[0])

    # 矩阵转置
    train_data = np.transpose(train_data, axes=[0, 2, 3, 1])
    test_data = np.transpose(test_data, axes=[0, 2, 3, 1])

    train_label = np.reshape(train_gt3D, [-1, train_gt3D.shape[1]*3])
    test_label = np.reshape(test_gt3D, [-1, test_gt3D.shape[1]*3])
    print(train_data.shape)

    # 模型路径的后缀
    model_path_suffix = '{}_{}_stage{}'.format(FLAGS.model_name, FLAGS.data_set, FLAGS.stages)
    # 模型保存路径
    model_save_dir = os.path.join(FLAGS.cacheDir,
                                  FLAGS.weightDir,
                                  model_path_suffix)
    # 训练和测试日志保存路径
    train_log_save_dir = os.path.join(FLAGS.cacheDir,
                                  FLAGS.logDir,
                                  model_path_suffix,
                                      'train')
    test_log_save_dir = os.path.join(FLAGS.cacheDir,
                                  FLAGS.logDir,
                                  model_path_suffix,
                                      'test')

    os.makedirs(model_save_dir,exist_ok=True)
    os.makedirs(train_log_save_dir,exist_ok=True)
    os.makedirs(test_log_save_dir,exist_ok=True)

    # 构建模型
    model = getModel()
    model.build_loss(weight_decay=FLAGS.weight_decay,
                     lr=FLAGS.init_lr,
                     lr_decay_rate=FLAGS.lr_decay_rate,
                     lr_decay_step=FLAGS.lr_decay_step)

    train_data_generator = Data_Generator(FLAGS.batch_size, images=train_data,
                                          labels=train_label, cubes=train_data_cube)
    test_data_generator = Data_Generator(FLAGS.batch_size, images=test_data,
                                          labels=test_label, cubes=test_data_cube)

    print("=====Model Build=====")

    merged_summary = tf.summary.merge_all()
    t1 = time.time()

    with tf.Session() as sess:
        # 创建 tensorboard
        train_writer = tf.summary.FileWriter(train_log_save_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_save_dir, sess.graph)

        # 创建 model saver
        saver = tf.train.Saver(max_to_keep=None)

        # 初始化 all vars
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 加载模型
        if FLAGS.pretrained_model != '':
            saver.restore(sess, os.path.join(model_save_dir, FLAGS.pretrained_model))
            print('load model from {}'.format(os.path.join(model_save_dir, FLAGS.pretrained_model)))
            # 检查权值
            for variable in tf.trainable_variables():
                with tf.variable_scope('', reuse=True):
                    var = tf.get_variable(variable.name.split(':0')[0])
                    print(variable.name, np.mean(sess.run(var)))

        global_step = 0
        while True:
            # 获取一个batch数据
            batch_x, batch_y, batch_cube = train_data_generator.next()

            train_ops = get_train_ops(model)
            # Forward and update weights
            # stage_losses_np, total_loss_np, _, current_lr, \
            # global_step, stage_errores_np, summaries = sess.run([model.stage_loss,
            #                                                      model.total_loss,
            #                                                      model.train_op,
            #                                                      model.cur_lr,
            #                                                      model.global_step,
            #                                                      model.stage_error,
            #                                                      merged_summary],
            #                                                     feed_dict={model.input_images: batch_x,
            #                                                                model.label_holder: batch_y,
            #                                                                model.cube_holder: batch_cube})
            train_ops.append(merged_summary)
            train_step_results = sess.run(train_ops,
                                          feed_dict={model.input_images: batch_x,
                                                     model.label_holder: batch_y,
                                                     model.cube_holder: batch_cube})
            summaries = train_step_results[-1]
            global_step = train_step_results[-2]
            train_writer.add_summary(summaries, global_step)
            # 打印训练中间过程
            if (global_step) % FLAGS.verbose_iters == 0:
                # Show training info
                print_current_training_stats(global_step, train_step_results, time.time() - t1)

            # 验证一下
            if (global_step) % FLAGS.validation_iters == 0:
                test_losses = []
                test_errors = []
                for _ in range(20):
                    batch_x, batch_y, batch_cube = test_data_generator.next()

                    eval_ops = get_eval_ops(model)
                    # test_batch_loss, test_batch_error, \
                    # summaries = sess.run([model.stage_loss[-1], model.stage_error[-1]
                    #                          , merged_summary],
                    #                      feed_dict={model.input_images: batch_x,
                    #                                 model.label_holder: batch_y,
                    #                                 model.cube_holder: batch_cube})
                    eval_ops.append(merged_summary)
                    test_batch_loss, test_batch_error, \
                    summaries = sess.run(eval_ops,
                                         feed_dict={model.input_images: batch_x,
                                                    model.label_holder: batch_y,
                                                    model.cube_holder: batch_cube})

                    test_losses.append(test_batch_loss)
                    test_errors.append(test_batch_error)

                test_mean_loss = np.mean(test_losses)
                test_mean_error = np.mean(test_errors)

                print('\n Validation loss:{}  Validation error:{}mm\n'.format(test_mean_loss, test_mean_error))
                test_writer.add_summary(summaries, global_step)

            # 保存模型
            if (global_step) % FLAGS.model_save_iters == 0:
                saver.save(sess=sess, global_step=global_step,
                           save_path=os.path.join(model_save_dir, FLAGS.model_name))
                print('\nModel checkpoint saved...\n')

            if global_step == FLAGS.training_iters:
                break

        print('Training done.')

def getModel():
    """
    根据FLAGS.network_def参数获取用于训练的模型
    :return: 特定结构的模型
    """
    if 'cpm' in FLAGS.network_def:
        m = model_module.CPM_Model(input_size=FLAGS.input_size,
                  joints=FLAGS.joints,
                  stages=FLAGS.stages,
                  input_type=FLAGS.INPUT_TYPE,
                  is_training=True)
    elif 'encoder' in FLAGS.network_def:
        m = model_module.Res_Encoder_Model(input_size=FLAGS.input_size,
                                    embed_size=FLAGS.embed,
                                    joints=FLAGS.joints,
                                    stacks=FLAGS.stacks,
                                    input_type=FLAGS.INPUT_TYPE,
                                    is_training=True)
    else:
        m = None
    return m

def get_train_ops(m):
    """
    获取训练过程中需要sess运行的操作（global_step 在列表最后）
    :param m: 操作来源模型
    :return: 操作列表
    """
    ops = []
    if 'cpm' in FLAGS.network_def:
        ops.append(m.stage_loss)
        ops.append(m.total_loss)
        ops.append(m.stage_error)
        ops.append(m.train_op)
        ops.append(m.cur_lr)
        ops.append(m.global_step)
    elif 'encoder' in FLAGS.network_def:
        ops.append(m.total_loss)
        ops.append(m.total_errors)
        ops.append(m.train_op)
        ops.append(m.cur_lr)
        ops.append(m.global_step)
    else:
        ops = None
    return ops

def get_eval_ops(m):
    """
    获取需要评估的操作，一般是loss和error
    :param m: 获取来源的模型
    :return: 操作列表
    """
    ops = []
    if 'cpm' in FLAGS.network_def:
        ops.append(m.stage_loss[-1])
        ops.append(m.stage_error[-1])
    elif 'encoder' in FLAGS.network_def:
        ops.append(m.total_loss)
        ops.append(m.total_errors)
    else:
        ops = None
    return ops

def print_current_training_stats(global_step, results, time_elapsed):
    """
    打印出目前的训练状态
    :param global_step: 当前的训练迭代次数
    :param results: sess运行后的结果
    :param time_elapsed: 运行时间
    :return:
    """
    if 'cpm' in FLAGS.network_def:
        stage_losses, total_loss, stage_errors, _, cur_lr, _, _ = results
        stats = 'Step: {} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step,
                                                                                  cur_lr, time_elapsed)
        losses = ' | '.join(
            ['S{} loss: {:>7.2f}'.format(stage_num + 1, stage_losses[stage_num]) for stage_num in
             range(len(stage_losses))])
        losses += ' | Total loss: {}'.format(total_loss)

        errors = ' | '.join(
            ['S{} error: {:>7.2f}mm'.format(stage_num + 1, stage_errors[stage_num]) for stage_num in
             range(len(stage_errors))])
        print(stats)
        print(losses)
        print(errors + '\n')
    elif 'encoder' in FLAGS.network_def:
        total_loss, total_errors, _, cur_lr, _, _ = results
        stats = 'Step: {} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step,
                                                                                  cur_lr, time_elapsed)
        losses = 'Total loss: {} | Total errors: {} mm'.format(total_loss, total_errors)
        print(stats)
        print(losses + '\n')

if __name__ == "__main__":
    tf.app.run()