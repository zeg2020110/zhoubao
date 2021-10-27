import tensorflow as tf
##报错信息来源于keras的导入，可以使用tensorflow.keras导人！
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans


##对测试图片进行规范，在尺寸和维度上和训练图片保持一致！
def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size) #skimage中的剪切函数，trans是上面导的包，
        if (not flag_multi_class):
            img = np.reshape(img,img.shape+(1,))  #升一个维度：（m,n,1）
        img = np.reshape(img,(1,)+img.shape) #再升一个维度：（1,m,n,1）----------有什么用呢？
        yield img



#对测试结果结果预测（分割）

#将分割的结果保存至指定路径！
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class,COLOR_DICT,item)
        else:
            img= item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)




def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)  # 输入函数，返回张量！
    # 这句话又是二合一了，可以等价的看为：
    '''
        #conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    '''
    # 64个卷积核，每个卷积核的尺寸为3*3
    # 先创建一个卷积层，这里的conv1表示一个函数。
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
    # 这里是向该卷积层输入张量inputs，conv1是其输出的值！
    conv1 = conv1(inputs)
    # 为什么要经过两次呢，提取高层次特征？
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # 2*2窗口的最大值池化！
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 我认为的第二层：
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 第三层
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 第四层
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # Dropout到底是怎样工作的？：就是对一个网络输出的张量进行丢弃，可以反映在网络中！
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # 第五层
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # 上采样，具体计算的过程没有找到，但结果就是将原来位置处的元素变为4个，都是自身！
    up5 = UpSampling2D(size=(2, 2))(drop5)
    # 第六层
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up5)
    # 将drop4与up6在第四个维度上进行拼接：即将两次卷积后的特征图拼接在一起！
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    # 第七层：
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    # 第八层：
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    # 第九层：
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # 第10层：
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    # 这里是使用keras中的函数式模型来搭建网络：只需使用inputs与outputs建立函数链式模型；
    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

model=unet()
testGene = testGenerator("data/test")
saver=tf.train.Saver()
init = tf.global_variables_initializer()#初始化
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './checkpoints/unet.cpkt')
    results = model.predict_generator(testGene, 30, verbose=1)#这里使用预测（predict）,是把标签图像看做是（0,1）的二分类问题吗！
    saveResult("data/result", results)

