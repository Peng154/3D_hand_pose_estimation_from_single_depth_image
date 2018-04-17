from model.model import Model
import tensorflow as tf

class ResEncoderModel(Model):
    def __init__(self, n_dim=63, embedSize=30, cacheFile=None):
        super(ResEncoderModel, self).__init__(cacheFile)
        self.n_dim = n_dim
        self.embedSize=embedSize
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
        nStages = [32, 64, 128, 256, 256]
        with tf.variable_scope('conv_1'):
            conv = self.lh.conv(X, filter_num=nStages[0], ksize=(5, 5), stride=1, reg=True)
            conv = self.lh.bn(conv)
            conv = self.lh.relu(conv)
            conv = self.lh.max_pool(conv, ksize=(2,2), stride=2)
        with tf.variable_scope('stack_1'):
            stack = self.lh.res_stack(conv, block_nums=5, filters_out=nStages[1], stride=2, reg=True, bottle_neck=True)
            # stack = self.lh.drop_out(stack, 0.5)
        with tf.variable_scope('stack_2'):
            stack = self.lh.res_stack(stack, block_nums=5, filters_out=nStages[2], stride=2, reg=True, bottle_neck=True)
            # stack = self.lh.drop_out(stack, 0.5)
        with tf.variable_scope('stack_3'):
            stack = self.lh.res_stack(stack, block_nums=5, filters_out=nStages[3], stride=2, reg=True, bottle_neck=True)
        with tf.variable_scope('stack_4'):
            stack = self.lh.res_stack(stack, block_nums=5, filters_out=nStages[4], stride=2, reg=True, bottle_neck=True)
        with tf.variable_scope('fc1'):
            fc = self.lh.fc(stack, 1024, reg=True)
            fc = self.lh.drop_out(fc, 0.7)
        with tf.variable_scope('fc2'):
            fc = self.lh.fc(fc, 1024, reg=True)
            fc = self.lh.drop_out(fc, 0.7)
        with tf.variable_scope('embed'):
            fc = self.lh.fc(fc, self.embedSize, reg=True)
        with tf.variable_scope('out'):
            fc = self.lh.fc(fc, self.n_dim, reg=True)

        loss = self.lh.sqr_loss(fc, y)

        self.loss = loss
        self.y_infer = fc

        return fc, loss