from src.Hand_Pose import HandPose
import tensorflow as tf
from src.util.handpose_evaluation import ICVLHandposeEvaluation
import pandas as pd
import numpy as np

TEST_BATCH = 1
MODEL_PATH = './train/train.ckpt'

def evaluate():

    handPose = HandPose('../data/ICVL')
    # 加载训练数据
    handPose.loadData()
    handPose.create_embedding()

    count = 0
    test_gt3D = [j.gt3Dorig for j in handPose.testSeqs[0].data]
    joints = []

    images = tf.placeholder(tf.float32, [i for i in handPose._test_data.shape])
    y = handPose.predict(images)

    # 将pca降维后的数据还原成48个骨骼点
    gt3D = handPose.add_Prior(y)
    batch = handPose.getNextBatch(TEST_BATCH)
    test_images_batch = batch[0]
    count += handPose.batchSize


    sess = tf.Session()
    if tf.gfile.Exists('./train/train.ckpt.meta'):
        print('加载模型。。。。')
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)
    else:
        print('未找到模型，停止测试。。。。')
        exit(1)

    predict_gt3D = gt3D.eval(feed_dict={images:test_images_batch}, session=sess)

    # pg = pd.DataFrame(predict_gt3D.reshape([702,-1]))
    # pg.to_csv('.test.csv1')

    for i in range(handPose.batchSize):
        index = i+handPose.batchSize*handPose.batch_index
        # print(index)
        joints.append(predict_gt3D[i].reshape((16, 3))*(handPose.testSeqs[0].config['cube'][2]/2.) +
                      handPose.testSeqs[0].data[index].com)

    # # 输出结果来看一看，蛋疼啊，都是一样的。。。
    # temp = np.asarray(joints)
    # temp = temp.reshape((702,-1))
    # pg = pd.DataFrame(temp)
    # pg.to_csv('.test.csv2')

    hpe = ICVLHandposeEvaluation(test_gt3D, joints)
    hpe.subfolder += '/ICVL_PCA30/'
    mean_error = hpe.getMeanError()
    print('my mean error:{} mm'.format(mean_error))

    #################################
    # BASELINE
    # Load the evaluation
    data_baseline = handPose.dataImporter.loadBaseline('../data/ICVL/LRF_Results_seq_1.txt')

    hpe_base = ICVLHandposeEvaluation(test_gt3D, data_baseline)
    hpe_base.subfolder += '/ICVL_PCA30/'
    print("Mean error: {}mm".format(hpe_base.getMeanError()))

    hpe.plotEvaluation('ICVL_PCA30', methodName='Our regr', baseline=[('Tang et al.', hpe_base)])

if __name__ == '__main__':
    evaluate()