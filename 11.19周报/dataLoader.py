import os
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg') #设置图像不显示在桌面上，但是这只是将显示终端改变了，图片其实还在！
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import pydicom   #pydicom库1.0以后没有dicom库了
from random import shuffle
import threading


##定义一个加载dicom数据的类！
class DCMDataLoader(object):
    #将__init__函数可以理解为java中的构造函数！
    def __init__(self, dcm_path, LDCT_image_path, NDCT_image_path, \
                 image_size=512, patch_size=64, depth=1, \
                 image_max=3072, image_min=-1024, batch_size=1, \
                 is_unpair=False, model='', num_threads=1):

        # dicom file dir
        self.dcm_path = dcm_path  #总路径
        self.LDCT_image_path = LDCT_image_path #低剂量CT路径
        self.NDCT_image_path = NDCT_image_path #全剂量CT路径

        # image params
        self.image_size = image_size #每一个切片的尺寸大小，这个数据集中确实是512*512的
        self.patch_size = patch_size #将每一张CT切片再分割成图像块
        self.depth = depth #图片块的通道数吧？

        self.image_max = image_max #设置像素点的有效范围？
        self.image_min = image_min

        # training params
        self.batch_size = batch_size #批量大小
        self.is_unpair = is_unpair  #不成对？什么意思？
        self.model = model   #是指VGG模型吗？

        # 创建两个数组，用来存放病人的CT切片图像名字，可是，内存够吗？不行还是先制作数据集吧！
        self.LDCT_image_name, self.NDCT_image_name = [], []

        # batch generator  prameters
        self.num_threads = num_threads #设置线程数目吗？
        self.capacity = 20 * self.num_threads * self.batch_size #设置容量，为下面创建对列时使用!
        self.min_queue = 10 * self.num_threads * self.batch_size #最小队列。又有什么用呢？

    # dicom file -> numpy array
    #可以理解为：在使用类创建对象时，可以直接通过对象执行此函数！
    def __call__(self, patent_no_list):
        p_LDCT = []
        p_NDCT = []
        for patent_no in patent_no_list:
            #获取每个病人的低剂量与全剂量切片的路径
            P_LDCT_path, p_NDCT_path = \
                glob(os.path.join(self.dcm_path, patent_no, self.LDCT_image_path, '*.dcm')), \
                glob(os.path.join(self.dcm_path, patent_no, self.NDCT_image_path, '*.dcm'))

            # load images
            #注意这列返回的图像是将切片堆叠在一起的矩阵！【本数据集中就是的维度就是（278,512,512）】
            org_LDCT_images, LDCT_slice_nm = self.get_pixels_hu(self.load_scan(P_LDCT_path),
                            '{}_{}'.format(patent_no, self.LDCT_image_path))#'{}_{}'相当于占位传值，传入的是病人的信息：patent_no_LDCT_image_path
            org_NDCT_images, NDCT_slice_nm = self.get_pixels_hu(self.load_scan(p_NDCT_path),
                                                                '{}_{}'.format(patent_no, self.NDCT_image_path))
            # CT slice name
            #这里是将多个病人的名字放到一个列表里
            self.LDCT_image_name.extend(LDCT_slice_nm)
            self.NDCT_image_name.extend(NDCT_slice_nm)

            # normalization
            #将归一化的CT图像存放到列表里
            p_LDCT.append(self.normalize(org_LDCT_images, self.image_max, self.image_min))
            p_NDCT.append(self.normalize(org_NDCT_images, self.image_max, self.image_min))

        #将所有病人的切片列表在第一个维度上进行和并！
        self.LDCT_images = np.concatenate(tuple(p_LDCT), axis=0)
        self.NDCT_images = np.concatenate(tuple(p_NDCT), axis=0)

        # image index
        #建立一个索引列表！注意这个len函数获取的是图像矩阵的第一个维度上的长度，即所有病人的数量！
        self.LDCT_index, self.NDCT_index = list(range(len(self.LDCT_images))), list(range(len(self.NDCT_images)))

    #加载CT图像【这里需要注意的是，一张CT图像由若干切片组成！】，返回的是一个切片列表吧
    def load_scan(self, path):
        slices = [pydicom.read_file(s) for s in path]#将所有的切片都加载进来
        #sort是对列表进行排序,lambda是隐函数，使用列表中的元素元组中第三个值进行排序！
        # ImagePositionPatient是一个三元double数组。用于表示当前图像坐标的原点（左上角）在参考坐标体系下的坐标。
        #ImagePositionPatient[2]是z轴上的坐标，
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        #异常处理：执行try语句，遇到异常，执行expect语句！
        try:
            slice_thickness = \
                np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])#计算两个切片图片的z轴坐标的差值！
        except:
            #SliceLocation获取层间距【即所在的Z轴位置。】
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            #SliceThickness用于表示相邻slice之间的距离；每个患者的情况不同，所以slice间的间距不同【是指每隔多少间（1mm）距取一个截面切片吗？】
            s.SliceThickness = slice_thickness
        return slices

    #将切片图像的或渎职转换为CT值的单位：Hounsfield，简称为Hu
    def get_pixels_hu(self, slices, pre_fix_nm):
        #s.pixel_array 将获取的dcm切片就转换成成了numpy数据！
        image = np.stack([s.pixel_array for s in slices]) #将所有切片堆叠在一起，相当于增加通道数！！
        image = image.astype(np.int16) #CT值的范围是-1024-3071，所以要用16位数据
        image[image == -2000] = 0

        digit = 3  #用来给切片命名的！我的这个数据集中只有是三位数！
        slice_nm = []
        for slice_number in range(len(slices)):
            #缩放斜率和截距由硬件制造商决定，它指定从存储在磁盘表示中的像素到存储在内存表示中的像素的线性转换。
            #磁盘存储的值定义为SV。而转化到内存中的像素值uints就需要两个dicom tag : Rescale intercept（b）和Rescale slope(m)：即OutputUnits=m∗SV+b
            intercept = slices[slice_number].RescaleIntercept#RescaleIntercept是CT图像的缩放截距
            slope = slices[slice_number].RescaleSlope#获取CT图像的缩放斜率
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float32)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)

            # sorted(idx), sorted(d_idx)  -> [1, 10, 2], [ 0001, 0002, 0010]
            s_idx = str(slice_number)
            #python中，字符串乘以数字等于几个同样的字符串！
            d_idx = '0' * (digit - len(s_idx)) + s_idx
            #将这些切片的名字存入列表！【也可以说是重命名了！】
            slice_nm.append(pre_fix_nm + '_' + d_idx)
        return np.array(image, dtype=np.int16), slice_nm

    #对CT图像进行正则化：
    def normalize(self, img, max_=3072, min_=-1024):
        img = img.astype(np.float32)
        #这个条件不是太懂？cycle是什么东西？
        if 'cycle' in self.model:  # -1 ~ 1
            img = 2 * ((img - min_) / (max_ - min_)) - 1 #最后归一化的范围是（-1,1）
            return img
        else:  # 0 ~ 1
            img = (img - min_) / (max_ - min_)
            return img

    #将输入的CT图片列表，切分成队列并返回！
    def input_pipeline(self, sess, image_size, end_point, depth=1):
        queue_input = tf.placeholder(tf.float32) #为输入提供占位符！
        queue_output = tf.placeholder(tf.float32)
        #创建一个队列元素按照先进先出的顺序出列的队列。
        queue = tf.FIFOQueue(capacity=self.capacity, dtypes=[tf.float32, tf.float32], \
                             # 这两个不是一样的尺寸吗？队列中的每一个元素是两个张图片组成的，所以需要设置两个图片尺寸！
                             shapes=[(image_size, image_size, depth), (image_size, image_size, depth)])
        #enqueue_many是一批张量的入列操作，将每个元素的第0维切分出来组成多个队列元素作为输入，输入的张量第0维的大小要相同
        enqueue_op = queue.enqueue_many([queue_input, queue_output])
        #close()用来关闭队列，标示没有元素再入列
        close_op = queue.close()
        #dequeue_many将n个元素连接到一起移出队列
        dequeue_op = queue.dequeue_many(self.batch_size)

        def enqueue(coord):
            enqueue_size = max(200, self.batch_size)
            if self.model == 'cyclegan':  # only cyclegan (cycelgain-identity:random patch))
                self.step = 0
                #来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，
                # 会抛出一个 OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程了;
                while not coord.should_stop():
                    start_pos = 0
                    if self.is_unpair:#数据不成对！
                        shuffle(self.LDCT_index)
                        shuffle(self.NDCT_index)
                    else:
                        self.NDCT_index = self.LDCT_index
                        shuffle(self.LDCT_index)

                    while start_pos < len(self.LDCT_index):
                        end_pos = start_pos + enqueue_size
                        #将整幅图像进行切割！（200,200）？还是有点大啊！
                        raw_LDCT_chunk = self.LDCT_images[self.LDCT_index][start_pos: end_pos]
                        raw_NDCT_chunk = self.NDCT_images[self.NDCT_index][start_pos: end_pos]

                        #将分割后的切片放入对列！
                        sess.run(enqueue_op, feed_dict={queue_input: np.expand_dims(raw_LDCT_chunk, axis=-1), \
                                                        queue_output: np.expand_dims(raw_NDCT_chunk, axis=-1)})
                        start_pos += enqueue_size
                    self.step += 1
                if self.step > end_point:
                    #request_stop()来发出终止所有线程的命令
                    coord.request_stop()
                sess.run(close_op)
            else:#与上面类似吧，只是不是cycle
                self.step = 0
                while not coord.should_stop():
                    LDCT_imgs, NDCT_imgs = [], []
                    for i in range(enqueue_size):
                        if self.is_unpair:
                            L_sltd_idx = np.random.choice(self.LDCT_index)
                            N_sltd_idx = np.random.choice(self.NDCT_index)
                        else:
                            L_sltd_idx = N_sltd_idx = np.random.choice(self.LDCT_index)

                        #在切片图像随意进行分割！
                        pat_LDCT, pat_NDCT = \
                            self.get_randam_patches(self.LDCT_images[L_sltd_idx],
                                                    self.NDCT_images[N_sltd_idx], image_size)
                        LDCT_imgs.append(np.expand_dims(pat_LDCT, axis=-1))
                        NDCT_imgs.append(np.expand_dims(pat_NDCT, axis=-1))
                    sess.run(enqueue_op, feed_dict={queue_input: np.array(LDCT_imgs), \
                                                    queue_output: np.array(NDCT_imgs)})
                    self.step += 1
                if self.step > end_point:
                    coord.request_stop()
                sess.run(close_op)

        self.coord = tf.train.Coordinator()#来创建一个线程协调器，用来管理之后在Session中启动的所有线程
        # threading.Thread() 创建子线程(target就是线程要执行的指定吧)
        self.enqueue_threads = [threading.Thread(target=enqueue, args=(self.coord,)) for i in range(self.num_threads)]
        for t in self.enqueue_threads: #上述线程对列，开始执行！
            t.start()

        return dequeue_op

    # WGAN_VGG, RED_CNN
    #将CT图像进行随机分割
    def get_randam_patches(self, LDCT_slice, NDCT_slice, patch_size, whole_size=512):
        whole_h = whole_w = whole_size
        h = w = patch_size

        # patch image range
        #np.round将数组四舍五入到给定的小数。
        hd, hu = h // 2, int(whole_h - np.round(h / 2))
        wd, wu = w // 2, int(whole_w - np.round(w / 2))

        # patch image center(coordinate on whole image)
        #选取切片分割块的起始点！
        h_pc, w_pc = np.random.choice(range(hd, hu + 1)), np.random.choice(range(wd, wu + 1))
        LDCT_patch = LDCT_slice[h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]
        NDCT_patch = NDCT_slice[h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]

        if self.model.lower() == 'red_cnn':
            return self.augumentation(LDCT_patch, NDCT_patch)
        return LDCT_patch, NDCT_patch

    # RED CNN
    #数据增强！
    def augumentation(self, LDCT, NDCT):
        """
        sltd_random_indx[0] :
            1: rotation
            2. flipping
            3. scaling
            4. pass
        sltd_random_indx[1] :
            select params
        """
        sltd_random_indx = [np.random.choice(range(4)), np.random.choice(range(2))]
        if sltd_random_indx[0] == 0:
            #旋转一个数组。是scipy.ndimage.interpolation的函数
            return rotate(LDCT, 45, reshape=False), rotate(NDCT, 45, reshape=False)
        elif sltd_random_indx[0] == 1:
            param = [True, False][sltd_random_indx[1]]
            if param:
                #[::-1]可以理解步长为-1，即将数组反转
                #[:,::-1]即表示对一个二维数组，第一维不变，第二维进行反转！
                return LDCT[:, ::-1], NDCT[:, ::-1]  # horizontal（水平翻转）
            return LDCT[::-1, :], NDCT[::-1, :]  # vertical（垂直翻转）
        elif sltd_random_indx[0] == 2:
            param = [0.5, 2][sltd_random_indx[1]]
            return LDCT * param, NDCT * param
        elif sltd_random_indx[0] == 3:
            return LDCT, NDCT



#计算峰值信噪比（psnr）
def tf_psnr(img1, img2, PIXEL_MAX=255.0):
    mse = tf.reduce_mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(PIXEL_MAX / tf.sqrt(mse))

# argparser string -> boolean type
#返回的是病人的什么？搞不懂？
def ParseList(l):
    return l.split(',')


# ROI crop
def ROI_img(whole_image, row=[200, 350], col=[75, 225]):
    patch_ = whole_image[row[0]:row[1], col[0]: col[1]]
    return np.array(patch_)


# psnr
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator





def psnr(img1, img2, PIXEL_MAX=255.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# save mk img
def save_image(LDCT, NDCT, output_, save_dir='.', max_=1, min_=0):
    f, axes = plt.subplots(2, 3, figsize=(30, 20))

    axes[0, 0].imshow(LDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 1].imshow(NDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 2].imshow(output_, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[1, 0].imshow(NDCT.astype(np.float32) - LDCT.astype(np.float32), cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 1].imshow(NDCT - output_, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 2].imshow(output_ - LDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[0, 0].title.set_text('LDCT image')
    axes[0, 1].title.set_text('NDCT image')
    axes[0, 2].title.set_text('output image')

    axes[1, 0].title.set_text('NDCT - LDCT  image')
    axes[1, 1].title.set_text('NDCT - outupt image')
    axes[1, 2].title.set_text('output - LDCT  image')
    if save_dir != '.':
        f.savefig(save_dir)
        plt.close()

    # ---------------------------------------------------


# argparser string -> boolean type
def ParseBoolean(b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError('Cannot parse string into boolean.')


