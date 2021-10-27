import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
import matplotlib.pylab as plt
import pickle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)



##定义真实图像与噪声图像
def get_inputs(real_size,noise_size): #定义图像的尺寸，图像是以行向量的形式来存放的，显示需要转化矩阵形式。
    real_img=tf.placeholder(tf.float32,[None,real_size],name='real_img')
    noise_img=tf.placeholder(tf.float32,[None,noise_size],name='noise_img')
    return real_img,noise_img

##定义生成器: 定义这玩意没啥用啊！
'''
class Generator(keras.models.Model): #这里相当于继承父类，括号里的是父类

    def __init__(self):  #此方法相当于java中类的构造函数；self是一个指向实例本身的引用，让实例可以访问类中方法和属性。
        super().__init__() #调用父类的构造函数，注意python2.7值与3之后的写法不同！

        # 定义的子类特有的属性
        self.n_f=512 #卷积核的个数相关
        self.n_k=4   #与卷积核的尺寸大小相关

        self.dense1=keras.layers.Dense(3*3*self.n_f) #全连接层：输出空间的维度为3*3*512.【这能做到吗？】
        #定义一个256个卷积核，卷积核的大小为3*3，步长为2，不填充的转置卷积层！
        self.conv2=keras.layers.Conv2DTranspose(self.n_f//2,3,2,'valid') #//表示整除并向下取整！
        self.bn2=keras.layers.BatchNormalization()#批量化标准层
        # 定义一个128个卷积核，卷积核的大小为4*4，步长为2，填充的转置卷积层！
        self.conv3=keras.layers.Conv2DTranspose(self.n_f//4,self.n_k,2,'same')
        self.bn3=keras.layers.BatchNormalization()
        # 定义一个1个卷积核，卷积核的大小为4*4，步长为2，填充的转置卷积层！
        self.conv4=keras.layers.Conv2DTranspose(1,self.n_k,2,'same')
        return

    def call(self,inputs,training=None):# 类中定义的方法比外面定义函数多了一个参数：self，指向实例本身的引用，让实例可以访问类中方法和属性。
        #计算Leaky ReLU激活函数；输入是一个张量
        x=tf.nn.leaky_relu(tf.reshape(self.dense1(inputs),shape=[-1,3,3,self.n_f])) #后面那个shape是reshape（）函数中的参数
        x=tf.nn.leaky_relu(self.bn2(self.conv2(x),name=training))
        x=tf.nn.leaky_relu(self.bn3(self.conv3(x),name=training))
        x=tf.tanh(self.conv4(x))
        return x
'''

##定义生成网络
def get_generator(noise_img,n_units,out_dim,reuse=None,alpha=0.01):
    #因为是两个网络分开训练，所以要将各自的参数放在自己的作用域中！
    with tf.variable_scope("generator",reuse=reuse): #创建一个名为"generator"的重用变量作用域，
        hidden1=tf.layers.dense(noise_img,n_units) #搭建全连接层
        #这里相当于是leaky ReLU函数的简单实现形式！alpha是其系数！
        hidden1=tf.maximum(alpha*hidden1,hidden1) #返回最大值？一个乘以0.01，还用比较吗？会出现负值吗？
        hidden1=tf.layers.dropout(hidden1,rate=0.2) #搭建dropout层
        logits=tf.layers.dense(hidden1,out_dim) #最终将输出的维数保持与真实图像一致，784
        outputs=tf.tanh(logits)
        return logits,outputs

##定义判别器
def get_discriminator(img,n_units,reuse=None,alpha=0.01):
    with tf.variable_scope("discriminator",reuse=reuse):
        hidden1=tf.layers.dense(img,n_units)
        hidden1=tf.maximum(alpha*hidden1,hidden1)
        logits=tf.layers.dense(hidden1,1)
        outputs=tf.sigmoid(logits)
        return logits,outputs

##定义相关参数
img_size=mnist.train.images[0].shape[0] #784=28*28
noise_size=100 #？
g_units=128 #设置生成器的全连接层的输出神经元个数。
d_units=128
alpha=0.01 #leaky_relu激活函数系数
learning_rate=0.001 #学习率
smooth=0.1 #？避免判决器的优化快于生成器？

tf.reset_default_graph #用于清除默认图形堆栈并重置全局默认图形。
real_img,noise_img=get_inputs(img_size,noise_size)
#g_outputs就是生成器生成的造假图像,g_logits是为了下面计算交叉熵使用的！
g_logits,g_outputs=get_generator(noise_img,g_units,img_size)#输入一张100*1的噪声图像，输出一张784*1的仿照图像。
#判决器分别对真实图像和假图像进行判断，变量重用！
d_logits_real,d_outputs_real=get_discriminator(real_img,d_units)
d_logits_fake,d_outputs_fake=get_discriminator(g_outputs,d_units,reuse=True)
#这里把真实图像的交叉熵（损失函数）乘上一个（1-smooth），是为了减小其梯度下降的速度吗？测试有没有都不影响结果。
d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real))*(1-smooth))
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_logits_fake)))
d_loss=tf.add(d_loss_real,d_loss_fake)
#生成器的损失函数采用假图像的交叉熵来定义，但是是与标签1的，因为生成器的目的就是让判决器识别为1.
g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_logits_fake))*(1-smooth))

'''
train_vars=tf.trainable_variables() #返回使用 trainable=True 创建的所有变量列表。

g_vars=[var for var in train_vars if var.name.startswith("generator")]

    等价于：
    g_vars=[]
    for var in train_vars ：
        if var.name.startswith("generator")]:
            g_vars.appends[var]

#d_vars=[var for var in train_vars if var.name.startswith("discriminator")] 写成这样应该效率更高点。
d_vars=[]
for var in train_vars :
    if var.name.startswith("discriminator"): #startswith是python中字符串的方法吗？
        d_vars.append(var)
'''
#新版本中可以直接区分两个域中定义的变量
g_vars=tf.trainable_variables("generator")
d_vars=tf.trainable_variables("discriminator")

d_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars)#使用var_list参数指明要更新的参数，使用判决器的loss当然只能更新它的变量。
g_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_vars)

##训练
batch_size=64 #批处理数量，一次取64张图片
epochs=300    #迭代300次。
n_sample=25
samples=[]    #存放生成器生成的假图片
losses=[]    #存放损失函数，下面要画图用。

#只保存生成器的变量吗？判决器的参数不用管它吗？确实不用，只要生成器训练好之后，直接批量造假就行，不需要在判决了，也就是或只要一个生成器就行了！
saver=tf.train.Saver(var_list=g_vars)
init = tf.compat.v1.global_variables_initializer()#初始化
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples//batch_size):#一次取64张，看看能取几次
            batch=mnist.train.next_batch(batch_size)#该函数就是每次批量获取图片。
            batch_images=batch[0].reshape((batch_size,784))
            #对图像的像素进行图像像素增长，这是因为tanh的输出结果介于（-1,1），真实图片和构造图片共享discriminator的参数
            batch_images=batch_images*2-1 #这样经过tanh的结果为（0,1）吗？

            batch_noise=np.random.uniform(-1,1,size=(batch_size,noise_size)) #随机生成（-1,1）之间的64*100的矩阵
            sess.run(d_train_opt,feed_dict={real_img:batch_images,noise_img:batch_noise})
            sess.run(g_train_opt,feed_dict={noise_img:batch_noise})
        train_loss_d=sess.run(d_loss,feed_dict={real_img:batch_images,noise_img:batch_noise})
        train_loss_d_real = sess.run(d_loss_real, feed_dict={real_img: batch_images, noise_img: batch_noise})
        train_loss_d_fake = sess.run(d_loss_fake, feed_dict={real_img: batch_images, noise_img: batch_noise})
        train_loss_g = sess.run(g_loss, feed_dict={noise_img: batch_noise})
        print("Epoch{}/{}...".format(e+1,epochs),"Discriminator Loss:{:.4f}(Real:{:.4f}+Fake:{:.4f}...".format(train_loss_d,train_loss_d_real,train_loss_d_fake),
              "Generator Loss:{:.4f}".format(train_loss_g))
        losses.append((train_loss_d,train_loss_d_real,train_loss_d_fake,train_loss_g))

        sample_noise=np.random.uniform(-1,1,size=(n_sample,noise_size)) #用来生成假图片的噪声
        gen_samples=sess.run(get_generator(noise_img,g_units,img_size,reuse=True),feed_dict={noise_img:sample_noise})
        samples.append(gen_samples)#将生成的假图片存放

        saver.save(sess,'./checkpoints/generator.cpkt')#保存的是整个模型吗？


#使用pickle模块保存生成的假图片样本
f = open('train_sample.pkl', 'wb')
train_sample = pickle.dump(samples, f, -1)
f.close()


##结果可视化
fig,ax=plt.subplots(figsize=(20,7)) #开一个大小为20*7的图像窗口！
losses=np.array(losses)
plt.plot(losses.T[0],label='Discriminator Total Loss') #对于一个数组，T[0]就是取出第一列的数据，
plt.plot(losses.T[1],label='Discriminator Real Loss')
plt.plot(losses.T[2],label='Discriminator Fake Loss')
plt.plot(losses.T[3],label='Generator Loss')
plt.title("Training Losses")
plt.legend()   #添加图例

#使用pickle模块读取数据
with open('train_sample.pkl', 'rb') as f:
    samples=pickle.load(f)
epoch_idx=[0,5,10,20,40,60,80,100,150,250]

show_imgs=[] #show_imgs是一个长度为10的列表，列表的每一个元素是一个长度为25的列表。
for i in epoch_idx:
    show_imgs.append(samples[i][1])  #这里取1是因为gen_samples的返回值是两个，1才是真正生成的对象

rows,cols=10,25 #10*25个子图
fig,axes=plt.subplots(figsize=(30,12),nrows=rows,ncols=cols,sharex=True,sharey=True)

idx=range(0,epochs,int(epochs/rows))
#axes本身就是一个（10,25）的数组，所以直接放入循环就行！
for sample,ax_row in zip(show_imgs,axes): #zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list（列表）。若传入参数的长度不等，则返回list的长度和参数中长度最短的对象相同
    for img,ax in zip(sample[::int(len(sample)/cols)],ax_row):
        ax.imshow(img.reshape((28,28)),cmap='Greys_r')
        ax.xaxis.set_visible(False) #每个子图都不不显示x轴
        ax.yaxis.set_visible(False)
plt.show()