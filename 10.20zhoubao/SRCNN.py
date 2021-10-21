import tensorflow as tf
import os
import cv2
import numpy as np

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


##设置占位符
x=tf.compat.v1.placeholder("float",[None,33,33,1])
y=tf.compat.v1.placeholder("float",[None,33,33,1])


##定义网络
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

filter2=[1,1,64,32]#64个（32通道）的5*5的卷积核
W_conv2=weight_variable(filter2)
b_conv2=bias_variable([32])
h_conv2=tf.nn.relu(conv2d(h_conv1,W_conv2)+b_conv2)

filter3=[5,5,32,1]#64个（32通道）的5*5的卷积核
b_conv3=bias_variable([1])
W_conv3=weight_variable(filter3)
y_output=conv2d(h_conv2,W_conv3)+b_conv3

##定义损失函数
loss=tf.reduce_mean(tf.reduce_mean(tf.square(y-y_output),axis=[1]))#这个损失函数有问题！
train_step=tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)
psnr=10.0*tf.log(255.0*255.0/loss)/tf.log(10.0)


##读取数据集
def read_TFRecord(filename):
    # filename是TFRecord文件路径，如果TFRecord和py文件在同一目录下可以只写文件名
    filename_queue = tf.train.string_input_producer([filename])
    # 创建一个读文件的对象
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # 得到一个记录，一定注意写的时候是int64解析的时候也要是int64
    features = tf.parse_single_example(serialized_example,
                        features={'label': tf.FixedLenFeature([], tf.string),
                                    'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    # 解码
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 必须重新指定shape，否则会报错，如果label也是一个数组，也必须指定shape
    img = tf.reshape(img, [33, 33, 1])
    # 获得标签
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [33, 33, 1])
    return img, label

##保存模型
saver=tf.train.Saver()



##训练
init = tf.compat.v1.global_variables_initializer()#初始化
with tf.compat.v1.Session() as sess:
    sess.run(init)
    i = 0
    img, label = read_TFRecord("train.tfrecords")
    # 使用shuffle_batch可以随机打乱输入，一批次8张图片
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=8,
                                                    capacity=8 * 64,
                                                    min_after_dequeue=8 * 32)

    threads = tf.train.start_queue_runners(sess=sess)
    # 输出五个批次的内容
    for i in range(2000):
        image, label = sess.run([img_batch, label_batch])
        sess.run(train_step, feed_dict={x:image,y:label})
        print("step %d :" %i)
        print(sess.run(psnr, feed_dict={x: image, y:label}))

        image=tf.cast(image,tf.float32)
        label = tf.cast(label, tf.float32)
        losss = tf.reduce_mean(tf.reduce_mean(tf.square(image - label), axis=[1]))
        print(sess.run(10.0 * tf.log(255.0 * 255.0 / losss) / tf.log(10.0)))

    saver.save(sess,'E:/program/tfTest/model/model3/model3.ckpt')#将整张图片分割成多张，来增加训练数量！