import tensorflow as tf
import numpy as np
from src.model.model import Model

class CPM_Model(Model):
    def __init__(self, input_size, joints, stages, input_type='DEPTH', is_training=True, cacheFile=None):
        super(CPM_Model, self).__init__(input_size, joints, input_type, is_training, cacheFile)
        self.stages = stages # stage的数目
        self.input_size = input_size
        self.joints = joints
        self.is_training = is_training
        self.stage_outputs = [] # 每个stage的关节点预测输出
        self.stage_feature_maps = []
        self.stage_error = [0 for _ in range(stages)]
        self.stage_loss = [0 for _ in range(stages)] # 每一个stage的loss
        self.total_loss = 0
        self.init_lr = 0
        self.weight_decay = 0.0

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

    def conv2d_bn(self, inputs, filters, kernel_size, strides, padding, activation, kernel_initializer, name):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(inputs=inputs,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    kernel_initializer=kernel_initializer,
                                    name='conv')
            bn = tf.layers.batch_normalization(conv, training=self.is_training, name='bn')
            out = activation(bn, name='activation')
            
            return out

    def _build_model(self):
        with tf.variable_scope('sub_stages'):
            sub_conv1 = tf.layers.conv2d(self.input_images,
                                         filters=64,
                                         kernel_size=[3,3],
                                         strides=[1,1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv1')
            sub_conv2 = tf.layers.conv2d(sub_conv1,
                                         filters=64,
                                         kernel_size=[3,3],
                                         strides=[1,1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv2')
            sub_pool1 = tf.layers.max_pooling2d(inputs=sub_conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='sub_pool1')
            sub_conv3 = tf.layers.conv2d(inputs=sub_pool1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv3')
            sub_conv4 = tf.layers.conv2d(inputs=sub_conv3,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv4')
            sub_pool2 = tf.layers.max_pooling2d(inputs=sub_conv4,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='sub_pool2')
            sub_conv5 = tf.layers.conv2d(inputs=sub_pool2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv5')
            sub_conv6 = tf.layers.conv2d(inputs=sub_conv5,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv6')
            sub_conv7 = tf.layers.conv2d(inputs=sub_conv6,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv7')
            sub_conv8 = tf.layers.conv2d(inputs=sub_conv7,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv8')
            sub_pool3 = tf.layers.max_pooling2d(inputs=sub_conv8,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='sub_pool3')
            sub_conv9 = tf.layers.conv2d(inputs=sub_pool3,
                                         filters=512,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='sub_conv9')
            sub_conv10 = tf.layers.conv2d(inputs=sub_conv9,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          name='sub_conv10')
            sub_conv11 = tf.layers.conv2d(inputs=sub_conv10,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          name='sub_conv11')
            sub_conv12 = tf.layers.conv2d(inputs=sub_conv11,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          name='sub_conv12')
            sub_conv13 = tf.layers.conv2d(inputs=sub_conv12,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          name='sub_conv13')
            sub_conv14 = tf.layers.conv2d(inputs=sub_conv13,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          name='sub_conv14')
            sub_pool4 = tf.layers.max_pooling2d(inputs=sub_conv14,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='sub_pool4')
            self.sub_stage_img_feature = tf.layers.conv2d(inputs=sub_pool4,
                                                          filters=128,
                                                          kernel_size=[3, 3],
                                                          strides=[1, 1],
                                                          padding='same',
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                          name='sub_stage_img_feature')

        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
                                     filters=512,
                                     kernel_size=[3, 3],
                                     strides=[1, 1],
                                     padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     name='conv1')
            # pool1 = tf.layers.max_pooling2d(inputs=conv1,
            #                                 pool_size=[2, 2],
            #                                 strides=2,
            #                                 padding='valid',
            #                                 name='pool1')
            # conv2 = tf.layers.conv2d(inputs=pool1,
            #                          filters=512,
            #                          kernel_size=[3, 3],
            #                          strides=[1, 1],
            #                          padding='same',
            #                          activation=tf.nn.relu,
            #                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            #                          name='conv2')

            stage_feature_map = tf.layers.conv2d(inputs=conv1,
                                     filters=self.joints * 3,
                                     kernel_size=[3, 3],
                                     strides=[1, 1],
                                     padding='same',
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     name='stage_feature_map')

            self.stage_feature_maps.append(stage_feature_map)

            output = tf.reduce_mean(stage_feature_map,axis=[1,2], name='stage_output')

            self.stage_outputs.append(output)

        for stage in range(2, self.stages+1):
            self.__middle_stage(stage_idx=stage)

    def __middle_stage(self, stage_idx):
        with tf.variable_scope('stage_'+str(stage_idx)):
            self.current_featuremap = tf.concat([self.stage_feature_maps[stage_idx-2],
                                                 self.sub_stage_img_feature],
                                                axis=3)
            mid_conv1 = tf.layers.conv2d(inputs=self.current_featuremap,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='mid_conv1')
            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='mid_conv2')
            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='mid_conv3')
            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='mid_conv4')
            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='mid_conv5')
            mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='valid',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                         name='mid_conv6')
            stage_feature_map = tf.layers.conv2d(inputs=mid_conv6,
                                                 filters=self.joints * 3,
                                                 kernel_size=[3, 3],
                                                 strides=[1, 1],
                                                 padding='same',
                                                 activation=None,
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                 name='stage_feature_map')
            self.stage_feature_maps.append(stage_feature_map)

            output = tf.reduce_mean(stage_feature_map, axis=[1, 2], name='stage_output')

            self.stage_outputs.append(output)

    @property
    def model_output(self):
        return self.stage_outputs[-1]

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

        for idx in range(self.stages):
            out_put = self.stage_outputs[idx]

            with tf.variable_scope('stage_'+ str(idx + 1)+'_loss'):
                self.stage_loss[idx] = tf.reduce_mean(tf.reduce_sum(tf.square(self.label_holder - out_put), axis=-1))
                tf.summary.scalar('stage_'+ str(idx + 1)+'_loss', self.stage_loss[idx])

            with tf.variable_scope('stage_'+ str(idx + 1)+'_error'):
                y_3Dcrop = tf.reshape(self.label_holder, [-1, self.label_holder.shape[1].value // 3, 3]) * (
                                tf.reshape(self.cube_holder, [-1, 1, 3]) / 2.)
                y_infer_3Dcrop = tf.reshape(out_put, [-1, out_put.shape[1].value // 3, 3]) * (
                                tf.reshape(self.cube_holder, [-1, 1, 3]) / 2.)
                # 计算误差
                errors = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(y_3Dcrop - y_infer_3Dcrop),
                                                             axis=-1)), axis=-1))  # 每个关节点的误差的和（mm）
                self.stage_error[idx] = errors
                tf.summary.scalar('stage_'+ str(idx + 1)+'_error', self.stage_error[idx])


        with tf.variable_scope('total_loss'):
            for idx in range(self.stages):
                self.total_loss += self.stage_loss[idx]
            self.total_loss += weight_decay * 0.5 * total_l2_loss
            tf.summary.scalar(name='total loss', tensor=self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()
            self.cur_lr = tf.train.exponential_decay(self.init_lr,
                                                     global_step=self.global_step,
                                                     decay_steps=self.lr_decay_step,
                                                     decay_rate=self.lr_decay_rate)
            tf.summary.scalar('global learning rate', self.cur_lr)
            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.cur_lr,
                                                            optimizer=self.optimizer)