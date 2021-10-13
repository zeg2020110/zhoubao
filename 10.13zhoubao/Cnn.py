import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

##读取数据【过时的写法】
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)#2.数据本身就是归一化的，也不会造成nan

##定义输入输出的占位
x=tf.compat.v1.placeholder("float",[None,784])
y=tf.compat.v1.placeholder("float",[None,10])

x_image=tf.reshape(x,[-1,28,28,1])#将图片集转为多张的28*28的单通道图片

##定义函数
def weight_variable(shape):#定义卷积核（网络之间）的权重：传入参数是张量的维度
    initial = tf.random.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):#定义偏差：传入参数是张量的维度
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):#定义卷积层：x是输入图像张量；W是卷积核
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#步长为1，补零，即保持维度。

def max_pool_2x2(x):#定义池化层：x是输入图像张量
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")#池化窗口为2*2，步长为2；补零

##构建卷积层和池化层的结构
#卷积层1:5*5
filter1=[5,5,1,32]#32个单通道的5*5的卷积核
W_conv1=weight_variable(filter1)
b_conv1=bias_variable([32])

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#这里相当32张的特征图像都加了b，使每张特征图像的每个像素值都加了0.1。这里的维数我很迷！
                                                    #激活函数是对每张特征图像的每个像素值进行激活运算的！
h_pool1=max_pool_2x2(h_conv1)
#卷积层2:5*5
filter2=[5,5,32,64]#64个（32通道）的5*5的卷积核
W_conv2=weight_variable(filter2)
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

h_pool2=max_pool_2x2(h_conv2)

##搭建全连接层
#全连接层1
W_fc1=weight_variable([7*7*64,1024])#这里定义的是一个全连接层的权重矩阵，输入神经元有7*7*64个，（需要将特征图展开，一个特征为3136*1的列向量），输出神经元为自定义：1024
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#-------------------------ERROR------------------------当设置keep_prob为1时，h_fc1_drop就会变为nan,使后面的一系列全为nan
keep_prob=tf.compat.v1.placeholder("float")#占位，定义dropout的比例
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)#可以将dropout看做（或者说就）是一个激活函数。
#-------------------------ERROR------------------------
#解决方案，使用tensorflow1.15版本！！


#--------------------------------------------------
#全连接层2
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

##训练模型
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)#输出的预测值；softmax也可以看做一个激活函数

#1.注意：这里的y_conv如果为0，进行对数运算时，无限大，会出现nan的情况，导致训练失败！这里我们在里面加上一个很小的数，防止这种情况！
cross_entropy=-tf.reduce_sum(y*tf.math.log(y_conv+ 1e-10))#定义损失函数，（交叉熵）#tf.math.log对数以e为底，向量中对应元素进行计算。#交叉熵定义的极为合理。
#cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step=tf.compat.v1.train.AdamOptimizer(0.0001).minimize(cross_entropy)#创建优化器，并设置最小化的损失函数
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))#argmax确定出估计的数值，equal判断是否正确！
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"double"))#这是计算所有样本的平均值（准确率）吗

##保存模型
saver=tf.train.Saver()

##运行
init = tf.compat.v1.global_variables_initializer()#初始化
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for i in range(20000):
        batch=mnist.train.next_batch(50)#每次获取50个图片和标签
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1],keep_prob:0.5})
        if i%100==0:
            train_accuracy=accuracy.eval(feed_dict={x:batch[0],y:batch[1],keep_prob:1})
            print("step %d,train accuracy %f"%(i,train_accuracy))
            print(sess.run(cross_entropy, feed_dict={x: batch[0], y: batch[1],keep_prob:1}))



    print("test accuracy %f"%accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1}))
    saver.save(sess,'E:/program/tfTest/model/model1/model1.ckpt')
