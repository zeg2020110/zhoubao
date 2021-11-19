import os
import tensorflow as tf
import numpy as np

##定义一些需要使用工具函数
#定义leaky_relu激活函数
def leaky_relu(x, leak=0.2):
    return tf.maximum(x, leak * x)

#定义卷积层
def conv2d(input, output_dim, ks=3, s=1, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return tf.keras.layers.Conv2D(output_dim, ks, s, padding=padding, activation=None)(input)

#定义全连接层
def fcn(input, n_weight, name):
    with tf.variable_scope(name):
        # 对于keras构建的全卷积网络，直接进行输入就行，不用将其展平！
        fc = tf.keras.layers.Dense(n_weight, activation=None)(input)
        return fc


##定义判决器网络
def discriminator(image, name="discriminator", reuse=True):
    #这种设置变量作用域的做法应该是静态图吧，tensorflow1.x版本
    with tf.variable_scope(name):#创建一个重用变量作用域，
        if reuse:
            #get_variable_scope()获取当前环境变量的作用域；
            # reuse_variables()设置作用域中变量的reuse参数
            tf.get_variable_scope().reuse_variables()

        l1 = leaky_relu(conv2d(image, 64, s=1, name='d_conv_1'))
        l2 = leaky_relu(conv2d(l1, 64, s=2, name='d_conv_2'))
        l3 = leaky_relu(conv2d(l2, 128, s=1, name='d_conv_3'))
        l4 = leaky_relu(conv2d(l3, 128, s=2, name='d_conv_4'))
        l5 = leaky_relu(conv2d(l4, 256, s=1, name='d_conv_5'))
        l6 = leaky_relu(conv2d(l5, 256, s=2, name='d_conv_6'))
        print(1111111111111111111111111111111)
        print(tf.shape(l6))
        fc1 = leaky_relu(fcn(l6, 1024, name='d_fc_1'))
        fc2 = fcn(fc1, 1, name='d_fc_2')

        return fc2


##定义生成器模型
def generator(image, name="generator", reuse=True):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        l1 = tf.nn.relu(conv2d(image, 32, name='g_conv_1'))
        l2 = tf.nn.relu(conv2d(l1, 32, name='g_conv_2'))
        l3 = tf.nn.relu(conv2d(l2, 32, name='g_conv_3'))
        l4 = tf.nn.relu(conv2d(l3, 32, name='g_conv_4'))
        l5 = tf.nn.relu(conv2d(l4, 32, name='g_conv_5'))
        l6 = tf.nn.relu(conv2d(l5, 32, name='g_conv_6'))
        l7 = tf.nn.relu(conv2d(l6, 32, name='g_conv_7'))
        l8 = tf.nn.relu(conv2d(l7, 1, name='g_conv_8'))

        return l8


class Vgg19:
    def __init__(self, size=64):
        self.size = size
        self.VGG_MEAN = [103.939, 116.779, 123.68]#这是什么，初始值？对图像做的一个变换

        vgg19_npy_path = "E:\\program\\PaperCode\\vgg19.npy"  #vgg19.npy是预训练的一个模型，可以拿来直接用的！
        #numpy.load()函数从具有npy扩展名(.npy)的磁盘文件返回输入数组
        #items() 函数是Python中字典(Dictionary)的函数，以列表返回可遍历的(键, 值) 元组数组。以列表返回可遍历的(键, 值) 元组数组。
        self.data_dict = np.load(vgg19_npy_path,  allow_pickle=True,encoding='latin1').item()#这里是获取模型Vgg19的参数
        print("npy file loaded")

    def extract_feature(self, rgb):
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        # split把张量分解为子张量
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        #get_shape()中返回tensor的形状，是一个元组（tuple）
        #可以使用 as_list()将元组转化为列表;[1:]是获取列表中第1个元素以后的元素，因为图片第一维是通道数！
        assert red.get_shape().as_list()[1:] == [self.size, self.size, 1]
        assert green.get_shape().as_list()[1:] == [self.size, self.size, 1]
        assert blue.get_shape().as_list()[1:] == [self.size, self.size, 1]
        #concat()是将tensor沿着指定维度连接起来,与concatenate等价
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        print(bgr.get_shape().as_list()[1:])
        assert bgr.get_shape().as_list()[1:] == [self.size, self.size, 3]

        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')
        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')
        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4, 'pool3')
        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.conv_layer(conv4_3, "conv4_4")
        pool4 = self.max_pool(conv4_4, 'pool4')
        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.conv_layer(conv5_3, "conv5_4")
        return conv5_4

    def conv_layer(self, bottom, name):#bottom是输入的图像
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")#这里是从vgg19.npy中取出参数。

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)










