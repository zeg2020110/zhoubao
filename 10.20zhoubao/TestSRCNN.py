import tensorflow as tf
import os
import cv2
import numpy as np

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

##设置占位符
x=tf.compat.v1.placeholder("float",[1,None,None,1])
y=tf.compat.v1.placeholder("float",[1,None,None,1])



def weight_variable(shape):#定义卷积核（网络之间）的权重：传入参数是张量的维度
    initial = tf.random.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):#定义偏差：传入参数是张量的维度
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):#定义卷积层：x是输入图像张量；W是卷积核
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#步长为1，补零，即保持维度。

filter1=[9,9,1,64]#32个单通道的5*5的卷积核
W_conv1=weight_variable(filter1)
b_conv1=bias_variable([64])
h_conv1=tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
h_conv1=tf.clip_by_value(h_conv1,0,255)

filter2=[1,1,64,32]#64个（32通道）的5*5的卷积核
W_conv2=weight_variable(filter2)
b_conv2=bias_variable([32])
h_conv2=tf.nn.relu(conv2d(h_conv1,W_conv2)+b_conv2)
h_conv2=tf.clip_by_value(h_conv2,0,255)

filter3=[5,5,32,1]#64个（32通道）的5*5的卷积核
b_conv3=bias_variable([1])
W_conv3=weight_variable(filter3)
y_output=conv2d(h_conv2,W_conv3)+b_conv3
y_output=tf.clip_by_value(y_output,0,255)


##定义损失函数
loss=tf.reduce_mean(tf.reduce_mean(tf.square(y-y_output),axis=[1]))#这个损失函数有问题！
#train_step=tf.compat.v1.train.AdamOptimizer(0.0001).minimize(loss)
psnr=10.0*tf.log(255.0*255.0/loss)/tf.log(10.0)


##对图片进行一些处理
def modcrop(image,scale = 2):
    h,w,= image.shape
    h = h - np.mod(h,scale)#计算余数
    w = w - np.mod(w,scale)
    image = image[0:h,0:w]
    return image
def updown(image):
    img1 = cv2.resize(image, None, fx=0.5, fy=0.5)
    img2 = cv2.resize(img1, None, fx=2, fy=2)
    return img2

saver=tf.train.Saver()

#对图片进行预处理



init = tf.compat.v1.global_variables_initializer()#初始化
with tf.compat.v1.Session() as sess:
    sess.run(init)
    saver.restore(sess, 'E:/program/tfTest/model/model4/model4.ckpt')

    image = cv2.imread('lenna.bmp', 0)
    input_y= modcrop(image, 2)
    input_x = updown(input_y)
    m,n=input_x.shape
    input_x.shape = (1, m, n, 1)
    input_y.shape = (1, m, n, 1)
    print(sess.run(psnr, feed_dict={x: input_x, y: input_y}))

    input_x = tf.cast(input_x, tf.float32)
    input_y = tf.cast(input_y, tf.float32)
    losss = tf.reduce_mean(tf.reduce_mean(tf.square(input_x-input_y), axis=[1]))
    print(sess.run(10.0*tf.log(255.0*255.0/losss)/tf.log(10.0)))
    '''
      之前代码有点问题，在对图片的预处理过程中，正反颠倒，所以效果不好，更正后整张训练的结果没有那么差
      在将测试集的一些数据放入之后，psnr得到提升，还有可能是数据集太少！29.216024------》32.200745
    '''

    '''
        将91张图片切割为33*33的若干小图片，训练后，效果为：32.34093--------------------》32.200745
    '''
    '''
        不填充的训练效果也不是太好啊，28.651638-------感觉还不如第二种呢！
    '''