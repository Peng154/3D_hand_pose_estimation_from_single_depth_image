from model.model import Model
import tensorflow as tf

class PoseRegModel(Model):
    def __init__(self, n_dim=30, cacheFile=None):
        super(PoseRegModel, self).__init__(cacheFile)
        self.n_dim = n_dim
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, n_dim], name='y')
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='X')

    def get_y_infer(self):
        if hasattr(self, 'y_infer'):
            return self.y_infer
        else:
            return None

    def get_loss(self):
        if hasattr(self, 'loss'):
            return self.loss
        else:
            return None

    def inference(self, X, y):
        with tf.variable_scope('conv1'):
            conv = self.lh.conv(X, filter_num=8, ksize=(5,5), stride=1, reg=True)
            pool = self.lh.max_pool(conv, ksize=(4,4), stride=4)
            relu = self.lh.relu(pool)
        with tf.variable_scope('conv2'):
            conv = self.lh.conv(relu, filter_num=8, ksize=(5,5),stride=1, reg=True)
            pool = self.lh.max_pool(conv, ksize=(2, 2), stride=2)
            relu = self.lh.relu(pool)
        with tf.variable_scope('conv3'):
            conv = self.lh.conv(relu, filter_num=8, ksize=(3,3),stride=1, reg=True)
            # pool = self.lh.max_pool(conv, ksize=(2, 2), stride=2)
            relu = self.lh.relu(conv)
        with tf.variable_scope('fc1'):
            fc = self.lh.fc(relu, 1024, reg=True)
            fc = self.lh.drop_out(fc, 0.7)
        with tf.variable_scope('fc2'):
            fc = self.lh.fc(fc, 1024, reg=True)
            fc = self.lh.drop_out(fc, 0.7)
        with tf.variable_scope('pca_out'):
            fc = self.lh.fc(fc, self.n_dim, reg=True)

        loss = self.lh.sqr_loss(fc, y)

        self.loss = loss
        self.y_infer = fc

        return fc, loss