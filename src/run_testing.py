import numpy as np
import tensorflow as tf
import os, importlib
import pickle

import matplotlib.pyplot as plt

from src.config import FLAGS
model_module = importlib.import_module('src.model.'+ FLAGS.network_def)
# from src.model.cpm_hand_model import CPM_Model

from data.importers import MSRA15Importer
from data.dataset import MSRA15Dataset

def main(argv):
    rng = np.random.RandomState(23455)

    print("create data")
    # aug_modes = ['com', 'rot', 'none']  # 'sc',

    comref = None  # "./eval/MSRA15_COM_AUGMENT/net_MSRA15_COM_AUGMENT.pkl"
    docom = False
    di = MSRA15Importer('../data/MSRA15/', refineNet=comref)

    # testSeqs = [di.loadSequence('Retrain', shuffle=True, rng=rng, docom=docom, cube=(200,200,200))]
    testSeqs = [di.loadSequence('P0', shuffle=True, rng=rng, docom=docom)]

    testDataSet = MSRA15Dataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly(testSeqs[0].name)
    test_data_cube = np.asarray([testSeqs[0].config['cube']] * test_data.shape[0])
    test_data = np.transpose(test_data, axes=[0, 2, 3, 1])

    test_label = np.reshape(test_gt3D, [-1, test_gt3D.shape[1] * 3])

    model = get_model()
    model.build_loss(weight_decay=FLAGS.weight_decay,
                     lr=FLAGS.init_lr,
                     lr_decay_rate=FLAGS.lr_decay_rate,
                     lr_decay_step=FLAGS.lr_decay_step)

    epoch_batch_num = test_data.shape[0] // FLAGS.batch_size

    model_path_suffix = '{}_{}_stage{}'.format(FLAGS.model_name, FLAGS.data_set, FLAGS.stages)
    # 测试模型参数加载路径
    model_weights_path = os.path.join(FLAGS.cacheDir,
                                      FLAGS.weightDir,
                                      model_path_suffix,
                                      '{}-{}'.format(FLAGS.model_name, FLAGS.test_iters))

    eval_result_dir = os.path.join(FLAGS.evalDir,
                                   model_path_suffix,
                                   'test_iter{}'.format(FLAGS.test_iters))
    os.makedirs(eval_result_dir, exist_ok=True)

    joint_errors = []
    total_errors = []
    losses = []


    with tf.Session() as sess:
        saver = tf.train.Saver()
        # print(model_weights_path)
        saver.restore(sess, model_weights_path)
        print('load model from: {}'.format(model_weights_path))

        joint_infer = tf.reshape(model.model_output, shape=[-1, model.model_output.shape[1].value//3, 3])* (
                                tf.reshape(model.cube_holder, [-1, 1, 3]) / 2.)
        joint_gt = tf.reshape(model.label_holder, [-1, model.label_holder.shape[1].value // 3, 3]) * (
                                tf.reshape(model.cube_holder, [-1, 1, 3]) / 2.)
        # batch中每个关节点的平均误差
        # joint_error_op = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(joint_infer - joint_gt), axis=-1)), axis=0)
        joint_error_op = tf.sqrt(tf.reduce_sum(tf.square(joint_infer - joint_gt), axis=-1))

        for i in range(epoch_batch_num):
            batch_x, batch_y, batch_cube = test_data[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size],\
                                           test_label[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], \
                                           test_data_cube[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size],

            ops = get_test_ops(model)
            ops.append(joint_error_op)
            loss, stage_error, joint_error = sess.run(ops,
                                                      feed_dict={model.input_images: batch_x,
                                                                model.label_holder: batch_y,
                                                                model.cube_holder: batch_cube})

            print('step {}, total loss {}, final stage error {} mm\n'.format(i, loss, stage_error))

            losses.append(loss)
            total_errors.append(stage_error)
            joint_errors.append(joint_error)

        print('average loss {}, average final stage error {} mm \n'.format(np.mean(losses), np.mean(total_errors)))

        # 保存评测误差
        pickle.dump(joint_errors, open('{}/joint_errors.pkl'.format(eval_result_dir), 'wb'))

        import matplotlib
        matplotlib.use('Agg') # 图像输出到文件
        draw_PCK(joint_errors, eval_result_dir)
        draw_joint_error(joint_errors, eval_result_dir)

def get_test_ops(m):
    """
    获取测试ops
    :param m: 来源模型
    :return:
    """
    ops = []
    if 'cpm' in FLAGS.network_def:
        # [model.stage_loss[-1], model.stage_error[-1]
        ops.append(m.stage_loss[-1])
        ops.append(m.stage_error[-1])
    elif 'encoder' in FLAGS.network_def:
        ops.append(m.total_loss)
        ops.append(m.total_errors)
    else:
        ops = None
    return ops

def get_model(weights_path=None):
    """
    构建测试模型
    :return:
    """
    if 'cpm' in FLAGS.network_def:
        m = model_module.CPM_Model(input_size=FLAGS.input_size,
                                joints=FLAGS.joints,
                                stages=FLAGS.stages,
                                input_type=FLAGS.INPUT_TYPE,
                                is_training=False,
                                cacheFile=weights_path)
    elif 'encoder' in FLAGS.network_def:
        m = model_module.Res_Encoder_Model(input_size=FLAGS.input_size,
                                    embed_size=FLAGS.embed,
                                    joints=FLAGS.joints,
                                    stacks=FLAGS.stacks,
                                    input_type=FLAGS.INPUT_TYPE,
                                    is_training=False,
                                    cacheFile=weights_path)
    else:
        m = None
    return m

def draw_PCK(joint_error_list, save_dir):
    """
    画出模型测试结果的PCK图像
    :param joint_error_list: 包含所有训练结果的关节误差列表，每个元素是一个batch的误差
    :param save_dir: PCK图像保存路径文件夹
    :return:
    """
    joint_errors = np.concatenate(joint_error_list, axis=0)
    pcks = []
    x = range(81)

    for threshold in x:
        mean_pck = np.mean(joint_errors < threshold)
        pcks.append(mean_pck)

    plt.figure(figsize=(8, 8))
    plt.grid(True)
    plt.plot(x, pcks)
    plt.xlabel('Distance threshold / mm')
    plt.ylabel('Fraction of frames within distance / %')

    file = '{}/pck.png'.format(save_dir)
    plt.savefig(file)
    print('save pck image:{}'.format(file))

def draw_joint_error(joint_error_list, save_dir):
    """
    画出模型每个关节点的平误差
    :param joint_error_list:  包含所有训练结果的关节误差列表，每个元素是一个batch的误差
    :param save_dir: 图像保存的文件夹路径
    :return:
    """
    array = np.concatenate(joint_error_list, axis=0)
    # print(array.shape)
    joint_mean_errors = np.mean(array, axis=0)
    print(joint_mean_errors.shape)
    x = np.arange(0, joint_mean_errors.shape[0])
    width = 0.5
    plt.figure(figsize=(8, 8))
    plt.xticks(x - width/2.0, ('P','I1','I2','I3','I4',
                               'M1','M2','M3','M4',
                               'R1','R2','R3','R4',
                               'L1','L2','L3','L4',
                               'T1','T2','T3','T4'))
    plt.bar(x - width/2.0, joint_mean_errors, width=width, align='center')
    file = '{}/joint_mean_error.png'.format(save_dir)
    plt.savefig(file)
    print('save joint_mean_error image:{}'.format(file))

if __name__ == '__main__':
    tf.app.run()