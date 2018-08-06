#-*- coding: utf_8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from cifar10 import *
import numpy as np
from attention_module import *
import time
import datetime
import numpy as np
import os
import argparse

   
def main(args):
    start = time.time()

    model_name = args.model_name
    log_path=os.path.join('logs',model_name)
    ckpt_path=os.path.join('model',model_name)
    if not os.path.exists(log_path):
      os.mkdir(log_path)
    if not os.path.exists(ckpt_path):
      os.mkdir(ckpt_path)

    weight_decay = args.weight_decay
    momentum = args.momentum
    init_learning_rate = args.learning_rate
    reduction_ratio = args.reduction_ratio
    batch_size = args.batch_size
    iteration = args.iteration
    # 128 * 391 ~ 50,000
    test_iteration = args.test_iteration
    total_epochs = args.total_epochs
    attention_module = args.attention_module
    
    def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
            if activation :
                network = Relu(network)
            return network

    def Fully_connected(x, units=class_num, layer_name='fully_connected') :
        with tf.name_scope(layer_name) :
            return tf.layers.dense(inputs=x, use_bias=True, units=units)

    def Relu(x):
        return tf.nn.relu(x)

    def Sigmoid(x):
        return tf.nn.sigmoid(x)

    def Global_Average_Pooling(x):
        return global_avg_pool(x, name='Global_avg_pooling')

    def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def Batch_Normalization(x, training, scope):
        with arg_scope([batch_norm],
                       scope=scope,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True) :
            return tf.cond(training,
                           lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda : batch_norm(inputs=x, is_training=training, reuse=True))

    def Concatenation(layers) :
        return tf.concat(layers, axis=3)

    def Dropout(x, rate, training) :
        return tf.layers.dropout(inputs=x, rate=rate, training=training)

    def Evaluate(sess):
        test_acc = 0.0
        test_loss = 0.0
        test_pre_index = 0
        add = 1000

        for it in range(test_iteration):
            test_batch_x = test_x[test_pre_index: test_pre_index + add]
            test_batch_y = test_y[test_pre_index: test_pre_index + add]
            test_pre_index = test_pre_index + add

            test_feed_dict = {
                x: test_batch_x,
                label: test_batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: False
            }

            loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

            test_loss += loss_
            test_acc += acc_

        test_loss /= test_iteration # average loss
        test_acc /= test_iteration # average accuracy

        summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                    tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

        return test_acc, test_loss, summary

    class SE_Inception_resnet_v2():
        def __init__(self, x, training):
            self.training = training
            self.model = self.Build_SEnet(x)

        def Stem(self, x, scope):
            with tf.name_scope(scope) :
                x = conv_layer(x, filter=32, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_conv1')
                x = conv_layer(x, filter=32, kernel=[3,3], padding='VALID', layer_name=scope+'_conv2')
                block_1 = conv_layer(x, filter=64, kernel=[3,3], layer_name=scope+'_conv3')

                split_max_x = Max_pooling(block_1)
                split_conv_x = conv_layer(block_1, filter=96, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')
                x = Concatenation([split_max_x,split_conv_x])

                split_conv_x1 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv2')
                split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv3')

                split_conv_x2 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv4')
                split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[7,1], layer_name=scope+'_split_conv5')
                split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[1,7], layer_name=scope+'_split_conv6')
                split_conv_x2 = conv_layer(split_conv_x2, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv7')

                x = Concatenation([split_conv_x1,split_conv_x2])

                split_conv_x = conv_layer(x, filter=192, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv8')
                split_max_x = Max_pooling(x)

                x = Concatenation([split_conv_x, split_max_x])

                x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                x = Relu(x)

                return x

        def Inception_resnet_A(self, x, scope):
            with tf.name_scope(scope) :
                init = x

                split_conv_x1 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv1')

                split_conv_x2 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv2')
                split_conv_x2 = conv_layer(split_conv_x2, filter=32, kernel=[3,3], layer_name=scope+'_split_conv3')

                split_conv_x3 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv4')
                split_conv_x3 = conv_layer(split_conv_x3, filter=48, kernel=[3,3], layer_name=scope+'_split_conv5')
                split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[3,3], layer_name=scope+'_split_conv6')

                x = Concatenation([split_conv_x1,split_conv_x2,split_conv_x3])
                x = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_final_conv1', activation=False)

                x = x*0.1

                # SE_block
                if attention_module == 'se_block':
                    x = se_block(x, scope+'_se_block', ratio=reduction_ratio)
                # CBAM_block
                if attention_module == 'cbam_block':
                    x = cbam_block(x, scope+'_cbam_block', ratio=reduction_ratio)


                x = init + x

                x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                x = Relu(x)

                return x

        def Inception_resnet_B(self, x, scope):
            with tf.name_scope(scope) :
                init = x

                split_conv_x1 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv1')

                split_conv_x2 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv2')
                split_conv_x2 = conv_layer(split_conv_x2, filter=160, kernel=[1,7], layer_name=scope+'_split_conv3')
                split_conv_x2 = conv_layer(split_conv_x2, filter=192, kernel=[7,1], layer_name=scope+'_split_conv4')

                x = Concatenation([split_conv_x1, split_conv_x2])
                x = conv_layer(x, filter=1152, kernel=[1,1], layer_name=scope+'_final_conv1', activation=False)
                # 1154
                x = x * 0.1

                # SE_block
                if attention_module == 'se_block':
                    x = se_block(x, scope+'_se_block', ratio=reduction_ratio)
                # CBAM_block
                if attention_module == 'cbam_block':
                    x = cbam_block(x, scope+'_cbam_block', ratio=reduction_ratio)

                x = init + x

                x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                x = Relu(x)

                return x

        def Inception_resnet_C(self, x, scope):
            with tf.name_scope(scope) :
                init = x

                split_conv_x1 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv1')

                split_conv_x2 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv2')
                split_conv_x2 = conv_layer(split_conv_x2, filter=224, kernel=[1, 3], layer_name=scope + '_split_conv3')
                split_conv_x2 = conv_layer(split_conv_x2, filter=256, kernel=[3, 1], layer_name=scope + '_split_conv4')

                x = Concatenation([split_conv_x1,split_conv_x2])
                x = conv_layer(x, filter=2144, kernel=[1,1], layer_name=scope+'_final_conv2', activation=False)
                # 2048
                x = x * 0.1

                # SE_block
                if attention_module == 'se_block':
                    x = se_block(x, scope+'_se_block', ratio=reduction_ratio)
                # CBAM_block
                if attention_module == 'cbam_block':
                    x = cbam_block(x, scope+'_cbam_block', ratio=reduction_ratio)

                x = init + x

                x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                x = Relu(x)

                return x

        def Reduction_A(self, x, scope):
            with tf.name_scope(scope) :
                k = 256
                l = 256
                m = 384
                n = 384

                split_max_x = Max_pooling(x)

                split_conv_x1 = conv_layer(x, filter=n, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')

                split_conv_x2 = conv_layer(x, filter=k, kernel=[1,1], layer_name=scope+'_split_conv2')
                split_conv_x2 = conv_layer(split_conv_x2, filter=l, kernel=[3,3], layer_name=scope+'_split_conv3')
                split_conv_x2 = conv_layer(split_conv_x2, filter=m, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

                x = Concatenation([split_max_x, split_conv_x1, split_conv_x2])

                x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                x = Relu(x)

                return x

        def Reduction_B(self, x, scope):
            with tf.name_scope(scope) :
                split_max_x = Max_pooling(x)

                split_conv_x1 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv1')
                split_conv_x1 = conv_layer(split_conv_x1, filter=384, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv2')

                split_conv_x2 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv3')
                split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

                split_conv_x3 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv5')
                split_conv_x3 = conv_layer(split_conv_x3, filter=288, kernel=[3,3], layer_name=scope+'_split_conv6')
                split_conv_x3 = conv_layer(split_conv_x3, filter=320, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv7')

                x = Concatenation([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])

                x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
                x = Relu(x)

                return x

        def Build_SEnet(self, input_x):
            input_x = tf.pad(input_x, [[0, 0], [32, 32], [32, 32], [0, 0]])
            # size 32 -> 96
            print(np.shape(input_x))
            # only cifar10 architecture

            x = self.Stem(input_x, scope='stem')

            for i in range(5) :
                x = self.Inception_resnet_A(x, scope='Inception_A'+str(i))

            x = self.Reduction_A(x, scope='Reduction_A')

            for i in range(10)  :
                x = self.Inception_resnet_B(x, scope='Inception_B'+str(i))

            x = self.Reduction_B(x, scope='Reduction_B')

            for i in range(5) :
                x = self.Inception_resnet_C(x, scope='Inception_C'+str(i))

            x = Global_Average_Pooling(x)
            x = Dropout(x, rate=0.2, training=self.training)
            x = flatten(x)

            x = Fully_connected(x, layer_name='final_fully_connected')
            return x

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)


    # image_size = 32, img_channels = 3, class_num = 10 in cifar10
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
    label = tf.placeholder(tf.float32, shape=[None, class_num])

    training_flag = tf.placeholder(tf.bool)


    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    logits = SE_Inception_resnet_v2(x, training=training_flag).model
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    train = optimizer.minimize(cost + l2_loss * weight_decay)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(log_path, sess.graph)

        epoch_learning_rate = init_learning_rate
        for epoch in range(1, total_epochs + 1):
            if epoch % 30 == 0 :
                epoch_learning_rate = epoch_learning_rate / 10

            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0

            for step in range(1, iteration + 1):
                if pre_index + batch_size < 50000:
                    batch_x = train_x[pre_index: pre_index + batch_size]
                    batch_y = train_y[pre_index: pre_index + batch_size]
                else:
                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]

                batch_x = data_augmentation(batch_x)

                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size


            train_loss /= iteration # average loss
            train_acc /= iteration # average accuracy

            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

            test_acc, test_loss, test_summary = Evaluate(sess)

            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()

            elapsed = time.time() - start
            elapsed_time = str(datetime.timedelta(seconds=elapsed))
            line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f, running_time: %s \n" % (epoch, total_epochs, train_loss, train_acc, test_loss, test_acc, elapsed_time)
            print(line)

            with open(os.path.join(log_path,'logs.txt'), 'a') as f:
                f.write(line)

            saver.save(sess=sess, save_path=os.path.join(ckpt_path,'Inception_resnet_v2.ckpt'))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name',   type=str, help='model name', default='model_temp')
    parser.add_argument('--attention_module', type=str, help='attention module name you want to use', default=None)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=0.0005)
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--learning_rate', type=float, help='learning_rate', default=0.1)
    parser.add_argument('--reduction_ratio', type=int, help='reduction_ratio', default=8)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=64)
    parser.add_argument('--iteration', type=int, help='training iteration', default=391)
    parser.add_argument('--test_iteration', type=int, help='test iteration', default=10)
    parser.add_argument('--total_epochs', type=int, help='total_epochs', default=100)
    
    return parser.parse_args(argv)
    
if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))