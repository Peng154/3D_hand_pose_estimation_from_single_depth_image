from src.Hand_Pose import HandPose
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # plot to file
import matplotlib.pyplot as plt
import pickle

TRAIN_BATCH = 0
TEST_BATCH = 1
MODEL_PATH = './train/train.ckpt'

def train():
    epochs = 250
    batchSize = 128
    weightDecay = 0.001
    learning_rate = 1e-3

    handPose = HandPose('../data/ICVL', weight_decay=weightDecay, batchSize=batchSize)
    # 加载训练数据
    handPose.loadData()
    handPose.create_embedding()

    shape = [handPose.batchSize]
    for i in range(1, len(handPose.train_data.shape)):
        shape.append(handPose.train_data.shape[i])
    print(shape)
    # 用于放置每一个batch的数据
    images = tf.placeholder(tf.float32, shape)
    y = handPose.predict(images)
    # 用于放置每一个batch的标记
    label = tf.placeholder(tf.float32, shape=[handPose.batchSize, 30])
    loss = handPose.loss(y, label)

    # 优化器使用RMSPropOptimizer,需要了解一下。。。
    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9, epsilon=1e-2).minimize(loss)

    # 觉得要配置一下GPU显存的的使用量
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)

    #用于记录所有的损失
    costs = []

    ##################################################
    # 开始训练:

    # 保存器
    saver = tf.train.Saver()
    last_step = -1

    if tf.gfile.Exists('./train/train.ckpt.meta'):
        print('已有模型，继续训练')
        f = open('./train/last_data.pkl', 'rb')
        # 加载上一次训练步数
        last_step, costs = pickle.load(f)
        f.close()
        saver.restore(sess, MODEL_PATH)
    else:
        print('未找到模型，开始训练')
        sess.run(tf.global_variables_initializer())

    current_step = last_step + 1

    # 如果没有找到训练的结果，就重新训练
    for i in range(last_step + 1, epochs):
        print("epochs:{}".format(i))
        current_step = i
        batch = handPose.getNextBatch(TRAIN_BATCH)
        while batch:
            train_images_batch = batch[0]
            train_gt3D_batch = batch[1]
            sess.run(train_step, feed_dict={images: train_images_batch, label: train_gt3D_batch})
            if (handPose.batch_index % 30 == 0):
                cost = loss.eval(feed_dict={images: train_images_batch, label: train_gt3D_batch}, session=sess)
                costs.append(cost)
                # if i== epochs-1:
                #     predict_embed = y.eval(feed_dict={images: train_images_batch}, session=sess)
                #     np.savetxt('train'+str(i)+','+str(handPose.batch_index)+'.csv', predict_embed, delimiter=',')
                print("epochs:{}, batches:{}, cost:{}".format(i, handPose.batch_index, cost))
            batch = handPose.getNextBatch(TRAIN_BATCH)

    # 保存训练次数以及所有的损失
    f = open('./train/last_data.pkl', 'wb')
    pickle.dump((current_step, costs), f)
    f.close()
    # 保存训练好的模型
    saver.save(sess=sess,save_path=MODEL_PATH)

    # 画出损失函数变化图像
    fig = plt.figure()
    plt.semilogy(costs)
    plt.show(block=False)
    fig.savefig('./costs.png')

def temp():
    f = open('./train/last_data.pkl', 'rb')
    # 加载上一次训练步数
    last_step, costs = pickle.load(f)
    last_step = 149
    f.close()

    f = open('./train/last_data.pkl', 'wb')
    pickle.dump((last_step, costs), f)
    f.close()


if __name__ =='__main__':
    train()
    # test()
