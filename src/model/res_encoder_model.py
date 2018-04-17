from src.model.model import Model
import tensorflow as tf
import numpy as np

class Res_Encoder_Model(Model):
    def __init__(self, input_size, embed_size, joints, stacks, input_type='DEPTH', is_training=True, cacheFile=None):
        super(Res_Encoder_Model, self).__init__(input_size, joints, input_type, is_training, cacheFile)

        self.stacks_num = stacks
        self.is_training = is_training
        self.joints = joints
        self.embed_size = embed_size
        self.output = None
        self.total_loss = 0
        self.total_errors = 0

        if input_type == 'DEPTH':
            self.input_images = tf.placeholder(dtype=tf.float32,
                                               shape=[None, input_size, input_size, 1],
                                               name='input_image_placeholder')
        elif input_type == 'RGB':
            self.input_images = tf.placeholder(dtype=tf.float32,
                                               shape=[None, input_size, input_size, 3],
                                               name='input_image_placeholder')

        self.label_holder = tf.placeholder(dtype=tf.float32, shape=[None, joints * 3])

        self.cube_holder = tf.placeholder(dtype=tf.float32, shape=[None, 3])

        self._build_model()

    def conv2d_bn(self, inputs, filters, kernel_size, strides, padding, kernel_initializer, name, activation=None,):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(inputs=inputs,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    kernel_initializer=kernel_initializer,
                                    name='conv')
            bn = tf.layers.batch_normalization(conv, training=self.is_training, name='bn')
            if(activation != None):
                out = activation(bn, name='activation')
            else:
                out = bn

            return out

    def res_stack(self, x, block_nums, filters_out, stride, name='res_stack', bottle_neck=True):
        assert block_nums >= 1
        out = x
        with tf.variable_scope(name):
            for i in range(block_nums):
                print('block_'+str(i))
                s = stride if i == 0 else [1, 1]  # 只有在每个stack开始的第一次卷积步长为输入，其他卷积层步长都为1
                with tf.variable_scope('bolck_{}'.format(i)):
                    out = self.res_block(out, filters_out, stride=s, bottle_neck=bottle_neck)
            return out

    def res_block(self, x, filters_out, stride, bottle_neck=True):
        filters_out_internal = filters_out // 4  # bottle_neck 中间卷积层的输出维数
        shortcut = x  # 高速连接层
        filters_in = x.get_shape()[-1].value

        if bottle_neck:
            x = self.conv2d_bn(x,
                               filters=filters_out_internal,
                               kernel_size=[1, 1],
                               strides=stride,
                               padding='same',
                               activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name='a')
            x = self.conv2d_bn(x,
                               filters=filters_out_internal,
                               kernel_size=[3, 3],
                               strides=[1, 1],
                               padding='same',
                               activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name='b')
            x = self.conv2d_bn(x,
                               filters=filters_out,
                               kernel_size=[1, 1],
                               strides=[1, 1],
                               padding='same',
                               activation=None,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name='c')
        else:
            x = self.conv2d_bn(x,
                               filters=filters_out,
                               kernel_size=[3, 3],
                               strides=stride,
                               padding='same',
                               activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name='A')
            x = self.conv2d_bn(x,
                               filters=filters_out,
                               kernel_size=[3, 3],
                               strides=[1, 1],
                               padding='same',
                               activation=None,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name='B')

        if filters_in != filters_out or stride != 1:
            shortcut = self.conv2d_bn(shortcut,
                               filters=filters_out,
                               kernel_size=[1, 1],
                               strides=stride,
                               padding='same',
                               activation=None,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name='shortcut')

        print(x.shape)
        return tf.nn.relu(x + shortcut)

    def flatten(self, x):
        input_dim = x.get_shape()
        fan_in = np.prod(input_dim[1:])
        if len(input_dim) > 2:  # 如果超过两维，就reshape
            x = tf.reshape(x, [-1, fan_in.value])
        return x

    def _build_model(self):
        nStages = [32]
        with tf.variable_scope('sub_stack'):
            sub_conv1 = self.conv2d_bn(self.input_images,
                                         filters=nStages[0],
                                         kernel_size=[5,5],
                                         strides=[1,1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv1')
            sub_stack_out = tf.layers.max_pooling2d(inputs=sub_conv1,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='sub_pool1')
        stack = sub_stack_out
        for idx in range(self.stacks_num):
            print('stack_' + str(idx))
            filters_out = nStages[0] * (2 ** (idx + 1))
            filters_out = filters_out if filters_out <= 256 else 256
            stack = self.res_stack(stack,
                                   block_nums=5,
                                   filters_out=filters_out,
                                   stride=[2, 2],
                                   name='stack_{}'.format(idx+1))

        with tf.variable_scope('dense_layers'):
            print(stack.shape)
            # 全连接层
            stack_out = self.flatten(stack)
            print(stack_out.shape)
            dense1 = tf.layers.dense(stack_out,
                                     units=1024,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     name='dense1')
            dense2 = tf.layers.dense(dense1,
                                     units=1024,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     name='dense2')
            embed = tf.layers.dense(dense2,
                                    units=self.embed_size,
                                    activation=None,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    name='embed')
            self.output = tf.layers.dense(embed,
                                  units=self.joints * 3,
                                  activation=None,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  name='out')

    @property
    def model_output(self):
        return self.output

    def build_loss(self, weight_decay, lr, lr_decay_rate, lr_decay_step, optimizer='Adam'):
        self.weight_decay = weight_decay
        self.init_lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.optimizer = optimizer

        total_l2_loss = 0
        for var in tf.global_variables():
            if 'kernel' in var.name:
                total_l2_loss += tf.reduce_sum(tf.square(var))

        with tf.variable_scope('total_error'):
            y_3Dcrop = tf.reshape(self.label_holder, [-1, self.label_holder.shape[1].value // 3, 3]) * (
                tf.reshape(self.cube_holder, [-1, 1, 3]) / 2.)
            y_infer_3Dcrop = tf.reshape(self.output, [-1, self.output.shape[1].value // 3, 3]) * (
                tf.reshape(self.cube_holder, [-1, 1, 3]) / 2.)
            # 计算误差
            self.total_errors = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(y_3Dcrop - y_infer_3Dcrop),
                                                                        axis=-1)), axis=-1))  # 每个关节点的误差的和（mm）
            tf.summary.scalar('total error', self.total_errors)

        with tf.variable_scope('total_loss'):
            self.total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.label_holder - self.output), axis=-1))
            self.total_loss += weight_decay * 0.5 * total_l2_loss # 正则化
            tf.summary.scalar(name='total loss', tensor=self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()
            self.cur_lr = tf.train.exponential_decay(self.init_lr,
                                                     global_step=self.global_step,
                                                     decay_steps=self.lr_decay_step,
                                                     decay_rate=self.lr_decay_rate)
            tf.summary.scalar('global learning rate', self.cur_lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.cur_lr,
                                                            optimizer=self.optimizer)