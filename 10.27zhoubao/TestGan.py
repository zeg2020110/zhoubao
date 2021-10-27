import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
import matplotlib.pylab as plt
import pickle
import numpy as np

#def get_inputs(real_size,noise_size): #定义图像的尺寸，图像是以行向量的形式来存放的，显示需要转化矩阵形式。
    #real_img=tf.placeholder(tf.float32,[None,real_size],name='real_img')
noise_img=tf.placeholder(tf.float32,[None,100],name='noise_img')
   # return real_img,noise_img

#def get_generator(noise_img,n_units,out_dim,reuse=None,alpha=0.01):
    #因为是两个网络分开训练，所以要将各自的参数放在自己的作用域中！
with tf.variable_scope("generator",reuse=None): #创建一个名为"generator"的重用变量作用域，
    hidden1=tf.layers.dense(noise_img,128) #搭建全连接层
    #这里相当于是leaky ReLU函数的简单实现形式！alpha是其系数！
    hidden1=tf.maximum(0.01*hidden1,hidden1) #返回最大值？一个乘以0.01，还用比较吗？会出现负值吗？
    hidden1=tf.layers.dropout(hidden1,rate=0.2) #搭建dropout层
    logits=tf.layers.dense(hidden1,784) #最终将输出的维数保持与真实图像一致，784
    outputs=tf.tanh(logits)
        #return logits,outputs

#g_vars=tf.trainable_variables("generator")
saver=tf.train.Saver()
samples=[]
#_,noise_img=get_inputs(784,100)
init = tf.global_variables_initializer()#初始化
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './checkpoints/generator.cpkt')
    sample_noise = np.random.uniform(-1, 1, size=(25, 100))  # 用来生成假图片的噪声
    gen_samples = sess.run(outputs, feed_dict={noise_img: sample_noise})
    samples.append(gen_samples)#将生成的假图片存放

f = open('test_sample.pkl', 'wb')
test_sample = pickle.dump(samples, f, -1)
f.close()


#使用pickle模块读取数据
with open('test_sample.pkl', 'rb') as f:
    samples=pickle.load(f)

rows,cols=5,5 #10*25个子图
fig,axes=plt.subplots(figsize=(30,12),nrows=rows,ncols=cols,sharex=True,sharey=True)
#sample是一个长度为25的列表，每个列表元素为一个图像
for sample,ax in zip(samples[0],axes): #zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list（列表）。若传入参数的长度不等，则返回list的长度和参数中长度最短的对象相同
    for i in range(len(ax)):
        ax[i].imshow(np.array(sample).reshape((28,28)),cmap='Greys_r')
        ax[i].xaxis.set_visible(False) #每个子图都不不显示x轴
        ax[i].yaxis.set_visible(False)
plt.show()