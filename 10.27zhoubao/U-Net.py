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


########################################首先对训练数据进行处理#################################################
#创建一个字典 ：用于定义增加训练数据（数据增强）的变换方式！
data_gen_args = dict(rotation_range=0.2,    #随机旋转的度数范围
                    width_shift_range=0.05, #宽度平移的范围？
                    height_shift_range=0.05,
                    shear_range=0.05,       #剪切强度（以弧度逆时针方向剪切角度
                    zoom_range=0.05,        #随机缩放范围
                    horizontal_flip=True,   #随机水平翻转
                    fill_mode='nearest')    #填充方法：最邻近插值


##对训练集的数据和标签的像素值进行归一化！
def adjustData(img, mask, flag_multi_class, num_class):
    if (flag_multi_class):  # 一个标签:如果是多类别的，再进行归一化操作！
        img = img / 255
        if (len(mask.shape) == 4):
            mask = mask[:, :, :, 0]  # 原来的4维数组为什么要少弄一维呢？
        else:
            mask = mask[:, :, 0]

        '''
            等价于：
            mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0] #这个要与切片操作区分，是取出矩阵的前几维度的元素。

        '''
        new_mask = np.zeros(mask.shape + (num_class,))  # 这里是（可以看做）两个元组相加，相加后原数组增长一个维度，为num_class！
        for i in range(num_class):  # 循环两次是因为新增的1个维度，有两个元素
            # 对于图像中的每一个像素，在mask中找到类并将其转化为一个热向量。
            '''
                one-hot向量是这样转换的，mask为0的位置，新建的newmask处的元素就为1【第一个维度】，mask为1的位置，新建的newmask处的元素也设为1【第二个维度】
                最终将多出的一个维度矩阵再变回去同维数的矩阵。
            '''
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            index = np.where(mask == i)  # 返回满足条件的索引坐标，使用元组表示；对于2维数组，坐标需要用2个数组表示，所以元组就是两个数组，两个数组组成满足条件的元素的坐标！
            if (len(mask.shape) == 4):
                index_mask = (
                index[0], index[1], index[2], np.zeros(len(index[0]), dtype=np.int64) + i)  # 给原来的索引元组增加1维，
            else:
                index_mask = (index[0], index[1], np.zeros(len(index[0]), dtype=np.int64) + i)
            new_mask[index_mask] = 1
            # 这个写法不是太明白，但是输出我明白了，mask == i返回一个与mask维度相同的矩阵；与后面i形成多一个维度的矩阵，将mask元素等于i的元素值替换为1。
            # 这一条语句顶上面这一大串。
            # new_mask[mask == i,i] = 1
        if (len(mask.shape) == 4):
            new_mask = np.reshape(new_mask,
                                  (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3]))
        else:
            new_mask = np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif (np.max(img) > 1):  ##清晰明了，进行归一化处理，mask变为二值图像（1,0）
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


##创建一个增加训练数据（数据增强）的图片生成器。
def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    '''
        参数：
        batch_size：批处理的数量；一次处理几张图片
        train_path：训练数据的路径；
        image_folder：图像文件夹的名称；
        mask_folder：标签文件夹的名称；
        aug_dict：变换字典，提供变换的一系列参数；
        image_color_mode/mask_color_mode：图片/标签的图片模式：灰度图/彩色图....
        image_save_prefix/mask_save_prefix ：生成的图片/标签的保存前缀（是指文件夹的吗？）
        flag_multi_class：多分类标签？
        num_class ：分类数量？
        save_to_dir ：保存路径；
        target_size：目标图像的尺寸；
        seed：种子数？
    '''

    '''
    can generate image and mask at the same time（可以同时生成图像与掩码【标签吧】）
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    （对生成的图像与掩码（标签）要使用相同的种子，以确保图像和掩码的转换是相同的！）
    if you want to visualize the results of generator, set save_to_dir = "your path"（可以保存并查看生成的图片！）
    '''
    image_datagen = ImageDataGenerator(**aug_dict)  # 这里是指用字典的形式传入参数！【**表示字典，*表示元组】
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    # train_generator =
    for (img, mask) in zip(image_generator, mask_generator):  # 并行遍历循环
        img, mask = adjustData(img, mask, flag_multi_class, num_class)  # 对生成的图像数据进行归一化，标签数据二值化。
        yield (img, mask)  # 简要理解：yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始。


myGene = trainGenerator(2,'data/train','image','label',data_gen_args,save_to_dir = None)

########################################定义U-Net网络###########################################
# 预训练权重：预训练的意思就是提前已经给你一些初始化的参数，这个参数不是随机的，而是通过其他类似数据集上面学得的，
# 然后再用你的数据集进行学习，得到适合你数据集的参数，--------这与迁移学习是不是有点关系啊！
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
saver=tf.train.Saver()
init = tf.compat.v1.global_variables_initializer()#初始化
with tf.compat.v1.Session() as sess:
    sess.run(init)
    #回调函数，将在每个epoch后保存模型到filepath
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    #使用 Python 生成器逐批生成的数据，按批次训练模型。可以在CPU 上对图像进行实时数据增强，以在 GPU 上训练模型。
    model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
    saver.save(sess,'./checkpoints/unet.cpkt')#保存的是整个模型吗？