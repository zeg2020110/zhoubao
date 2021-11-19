import os
import tensorflow as tf
import numpy as np
import time
from glob import glob
import GanVgg as modules
import dataLoader as ut


class wganVgg(object):
    def __init__(self, sess, dcm_path, test_patient_no, checkpoint_dir, \
                 LDCT_path, NDCT_path, whole_size, patch_size, img_channel, img_vmax, img_vmin, \
                 batch_size, model, phase, num_iter, lambda_, lambda_1, alpha, beta1, beta2, \
                 ):  # arg应该是命令中输入的参数列表
        ######注意：类与函数的区别，函数传递的参数可以直接在函数中使用 ，类初始化是传递的参数要想使用，全局需要使用属性去承接！！！#############
        self.sess = sess  # 这个sess是tensorflow计算时开的一个会话
        self.dcm_path = dcm_path
        self.checkpoint_dir = checkpoint_dir
        self.LDCT_path = LDCT_path
        self.NDCT_path = NDCT_path
        self.whole_size = whole_size
        self.patch_size = patch_size
        self.img_channel = img_channel
        self.img_vmax = img_vmax
        self.img_vmin = img_vmin
        self.batch_size = batch_size
        self.model=model



        ####patients folder name
        '''
         self.train_patient_no = [d.split('\\')[-1] for d in glob(dcm_path + '/*') if
                                 ('zip' not in d) & (d.split('\\')[-1] not in test_patient_no)]

        '''

        self.test_patient_no = test_patient_no
        ds = glob(dcm_path + '/*')
        train_patient_no = []
        for d in ds:
            if ('zip' not in d) & (d.split('\\')[-1] not in test_patient_no):
                train_patient_no.append(d.split('\\')[-1])
        print(len(train_patient_no))
        self.train_patient_no = train_patient_no

        # save directory
        # join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。
        self.p_info = '_'.join(self.test_patient_no)
        self.checkpoint_dir = os.path.join('', checkpoint_dir, self.p_info)
        self.log_dir = os.path.join('', 'logs', self.p_info)
        print('directory check!!\ncheckpoint : {}\ntensorboard_logs : {}'.format(self.checkpoint_dir, self.log_dir))

        #### set modules (generator, discriminator, vgg net)
        self.g_net = modules.generator
        self.d_net = modules.discriminator
        self.vgg = modules.Vgg19()

        """
        load images
        """

        print('data load... dicom -> numpy')
        '''

        '''
        # self.image_loader = dl.DCMDataLoader(dcm_path, LDCT_image_path=LDCT_path, NDCT_image_path=NDCT_path, model=model, patent_no_list=train_patient_no)
        self.image_loader = ut.DCMDataLoader(self.dcm_path, self.LDCT_path, self.NDCT_path, \
                                             image_size=self.whole_size, patch_size=self.patch_size,
                                             depth=self.img_channel,
                                             image_max=self.img_vmax, image_min=self.img_vmin,
                                             batch_size=self.batch_size, model=self.model)
        self.test_image_loader = ut.DCMDataLoader(self.dcm_path, self.LDCT_path, self.NDCT_path, \
                                                  image_size=self.whole_size, patch_size=self.patch_size,
                                                  depth=self.img_channel,
                                                  image_max=self.img_vmax, image_min=self.img_vmin,
                                                  batch_size=self.batch_size, model=self.model)

        # Python time time() 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
        t1 = time.time()
        if phase == 'train':
            # 将所有训练病人的切片的切片列表在第一个列表上合并！
            self.image_loader(self.train_patient_no)
            self.test_image_loader(self.test_patient_no)
            print('data load complete !!!, {}\nN_train : {}, N_test : {}'.format(time.time() - t1,
                                                                                 len(self.image_loader.LDCT_image_name),
                                                                                 len(
                                                                                     self.test_image_loader.LDCT_image_name)))
            [self.z_i, self.x_i] = self.image_loader.input_pipeline(self.sess, patch_size, num_iter)
        else:  # 进行测试
            self.test_image_loader(self.test_patient_no)
            print('data load complete !!!, {}, N_test : {}'.format(time.time() - t1,
                                                                   len(self.test_image_loader.LDCT_image_name)))
            self.z_i = tf.placeholder(tf.float32, [None, patch_size, patch_size, img_channel],
                                      name='whole_LDCT')
            self.x_i = tf.placeholder(tf.float32, [None, patch_size, patch_size, img_channel],
                                      name='whole_LDCT')

        """
        build model
        """
        #### image placehold  (patch image, whole image)
        self.whole_z = tf.placeholder(tf.float32, [1, whole_size, whole_size, img_channel],
                                      name='whole_LDCT')
        self.whole_x = tf.placeholder(tf.float32, [1, whole_size, whole_size, img_channel],
                                      name='whole_NDCT')

        #### generate & discriminate
        # generated images
        # 往生成器中输入图片，
        self.G_zi = self.g_net(self.z_i, reuse=False)  # 输出切片分割图像
        self.G_whole_zi = self.g_net(self.whole_z)  # 整张输入！

        # discriminate
        # 往判决器中输入图片
        self.D_xi = self.d_net(self.x_i, reuse=False)
        self.D_G_zi = self.d_net(self.G_zi)

        #### loss define
        # gradients penalty
        # 这是Wgan中的梯度惩罚，用来防止Gan网络在训练过程中梯度爆炸与消失！
        self.epsilon = tf.random_uniform([], 0.0, 1.0)  # 生成随机均匀分布的参数。
        # 这里为什么要将生成器生成的图像与原始低剂量图像叠加后再送入判决器呢？
        self.x_hat = self.epsilon * self.x_i + (1 - self.epsilon) * self.G_zi
        self.D_x_hat = self.d_net(self.x_hat)
        # 实现判决结果对生成图像的求导结果！提到了梯度与自变量维度一致【返回的结果是对x_hat的多个变量的求导结果】[0]表示只取对一个变量的偏导
        self.grad_x_hat = tf.gradients(self.D_x_hat, self.x_hat)[0]  # grad_x_hat得到的是三维数组，因为x_hat是四维数组，【0】为取第一维度索引为0的元素
        # reduce_sum对数组按指定的轴进行求和，当为二维数组时，axis=1即按行求和，
        self.grad_x_hat_l2 = tf.sqrt(tf.reduce_sum(tf.square(self.grad_x_hat), axis=1))  # 那个这个变量为2维数组
        self.gradient_penalty = tf.square(self.grad_x_hat_l2 - 1.0)

        # perceptual loss【这就是所谓的感知损失！】
        # 将3个数组在axis=3的维数进行拼接！
        self.G_zi_3c = tf.concat([self.G_zi] * 3, axis=3)
        self.xi_3c = tf.concat([self.x_i] * 3, axis=3)
        [w, h, d] = self.G_zi_3c.get_shape().as_list()[1:]
        self.vgg_perc_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
            # 把低剂量图像与生成后的图像送入VGG有什么用吗?
            tf.square((self.vgg.extract_feature(self.G_zi_3c) - self.vgg.extract_feature(self.xi_3c))))) / (w * h * d))

        # discriminator loss(WGAN LOSS)
        # 判决器对生成图像的判决与低剂量图像的判决结果之差！
        d_loss = tf.reduce_mean(self.D_G_zi) - tf.reduce_mean(self.D_xi)
        # 梯度惩罚项！
        grad_penal = lambda_ * tf.reduce_mean(self.gradient_penalty)
        self.D_loss = d_loss + grad_penal
        # generator loss
        # 感知损失减去判决器对生成图片的判决结果
        self.G_loss = lambda_1 * self.vgg_perc_loss - tf.reduce_mean(self.D_G_zi)

        #### variable list
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        """
        summary
        """
        # loss summary
        # tf.summary.scalar用来显示标量摘要！
        self.summary_vgg_perc_loss = tf.summary.scalar("1_PerceptualLoss_VGG", self.vgg_perc_loss)
        self.summary_d_loss_all = tf.summary.scalar("2_DiscriminatorLoss_WGAN", self.D_loss)
        self.summary_d_loss_1 = tf.summary.scalar("3_D_loss_disc", d_loss)
        self.summary_d_loss_2 = tf.summary.scalar("4_D_loss_gradient_penalty", grad_penal)
        self.summary_g_loss = tf.summary.scalar("GeneratorLoss", self.G_loss)
        # tf.summary.merge：合并summaries，创建一个summary协议缓冲区，包含输入的summaries的所有value
        self.summary_all_loss = tf.summary.merge(
            [self.summary_vgg_perc_loss, self.summary_d_loss_all, self.summary_d_loss_1, self.summary_d_loss_2,
             self.summary_g_loss])

        # psnr summary
        # 计算原低剂量图像的psnr,(金标准为全剂量图像)
        self.summary_psnr_ldct = tf.summary.scalar("1_psnr_LDCT", ut.tf_psnr(self.whole_z, self.whole_x, 1),
                                                   family='PSNR')  # 0 ~ 1
        # 计算低剂量生成图像的psnr,(金标准为全剂量图像)-----------参数是不是反了？
        self.summary_psnr_result = tf.summary.scalar("2_psnr_output", ut.tf_psnr(self.whole_x, self.G_whole_zi, 1),
                                                     family='PSNR')  # 0 ~ 1
        self.summary_psnr = tf.summary.merge([self.summary_psnr_ldct, self.summary_psnr_result])

        # image summary
        # z_i,x_i为输入的切片队列，G_zi为生成的低剂量去噪图片，这些矩阵原来为4维，取一个三维矩阵再进行扩维，再连接。
        self.check_img_summary = tf.concat([tf.expand_dims(self.z_i[0], axis=0), \
                                            tf.expand_dims(self.x_i[0], axis=0), \
                                            tf.expand_dims(self.G_zi[0], axis=0)], axis=2)  # 在第三维上进行连接，这里就有点奇怪了啊！
        # 用来输出Summary的图像
        self.summary_train_image = tf.summary.image('0_train_image', self.check_img_summary)
        # 将这些图像在axis=2维度进行连接，有什么用呢？【显示的还对吗？】
        self.whole_img_summary = tf.concat([self.whole_z, self.whole_x, self.G_whole_zi], axis=2)
        self.summary_image = tf.summary.image('1_whole_image', self.whole_img_summary)

        #### optimizer
        self.d_adam, self.g_adam = None, None
        # tf.control_dependencies()函数是用来控制计算流图的，也就是给图中的某些计算指定顺序：先更新判决器，再更新生成器！
        # 该函数可以用来获取key集合中的所有元素，返回一个列表。列表的顺序依变量放入集合中的先后而定。
        # 关于tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作，配合tf.control_dependencies函数使用
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1, beta2=beta2).minimize(
                self.D_loss, var_list=self.d_vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1, beta2=beta2).minimize(
                self.G_loss, var_list=self.g_vars)

        # model saver
        self.saver = tf.train.Saver(max_to_keep=None)

        print('--------------------------------------------\n# of parameters : {} '. \
              # np.prod()函数用来计算所有元素的乘积
              # 返回使用 trainable=True 创建的所有变量列表。
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))  # 计算是模型的参数个数！

    def train(self, continue_train, num_iter, d_iters, print_freq, save_freq, model):
        self.sess.run(tf.global_variables_initializer())  # 初始化全局变量
        # 将训练日志写入到logs文件夹下；参数sess.graph是事件文件要记录的图，也就是TensorFlow默认的图。
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.start_step = 0
        if continue_train:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        print('Start point : iter : {}'.format(self.start_step))

        start_time = time.time()

        for t in range(self.start_step, num_iter):  # 外循环训练生成器
            for _ in range(0, d_iters):  # 这是先训练判决器，（内循环训练判决器）
                # discriminator update
                self.sess.run(self.d_adam)  # 这一步就是进行训练吧！

            # generator update & loss summary
            _, summary_str = self.sess.run([self.g_adam, self.summary_all_loss])
            self.writer.add_summary(summary_str, t)  # 将训练过程数据保存在filewriter指定的文件中
            # print point
            if (t + 1) % print_freq == 0:  # 这是每隔多少次，将训练结果（图片）查看一下，并输出误差！
                # print loss & time
                d_loss, g_loss, g_zi_img, summary_str0 = self.sess.run(
                    [self.D_loss, self.G_loss, self.G_zi, self.summary_train_image])
                # training sample check
                self.writer.add_summary(summary_str0, t)  # 将训练过程中的图像加入到日志中！

                print('Iter {} Time {} d_loss {} g_loss {}'.format(t, time.time() - start_time, d_loss, g_loss))
                self.check_sample(t)

            if (t + 1) % save_freq == 0:  # 每隔多少次，保存一下模型！
                self.save(model, t)

        self.image_loader.coord.request_stop()  # 关闭tf的多个线程
        # 使用coord.join(threads)把线程加入主线程，等待threads结束。
        self.image_loader.coord.join(self.image_loader.enqueue_threads)

    def load(self):
        print(" [*] Reading checkpoint...")
        # 通过checkpoint文件找到模型文件名。
        # 该函数返回的是checkpoint文件CheckpointState proto类型的内容，其中有model_checkpoint_path和
        # all_model_checkpoint_paths两个属性。其中model_checkpoint_path保存了最新的tensorflow模型文件的文件名，
        # all_model_checkpoint_paths则有未被删除的所有tensorflow模型文件的文件名。
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # basename返回path最后的文件名
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            # restore是加载模型！
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            print(self.start_step)
            return True
        else:
            return False

    # summary test sample image during training
    def check_sample(self, t):
        # summary whole image'
        # np.random.choice：从数组中随机抽取元素，
        sltd_idx = np.random.choice(range(len(self.test_image_loader.LDCT_images)))
        # 在测试集中选取一组测试图片
        test_zi, test_xi = self.test_image_loader.LDCT_images[sltd_idx], self.test_image_loader.NDCT_images[
            sltd_idx]
        # 将测试图像输入生成器进行生成图像！
        whole_G_zi = self.sess.run(self.G_whole_zi,
                                   feed_dict={self.whole_z: test_zi.reshape(self.whole_z.get_shape().as_list())})

        summary_str1, summary_str2 = self.sess.run([self.summary_image, self.summary_psnr], \
                                                   feed_dict={self.whole_z: test_zi.reshape(
                                                       self.whole_z.get_shape().as_list()), \
                                                              self.whole_x: test_xi.reshape(
                                                                  self.whole_x.get_shape().as_list()), \
                                                              self.G_whole_zi: whole_G_zi.reshape(
                                                                  self.G_whole_zi.get_shape().as_list()), \
                                                              })
        self.writer.add_summary(summary_str1, t)  # 将测试的图像保存在filewriter指定的文件中
        self.writer.add_summary(summary_str2, t)  # 将测试的psnr保存在filewriter指定的文件中

    def save(self, model, step):
        model_name = model + ".model"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def test(self, test_npy_save_dir):
        self.sess.run(tf.global_variables_initializer())

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## mk save dir (image & numpy file)
        npy_save_dir = os.path.join('', test_npy_save_dir, self.p_info)

        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)

        ## test
        start_time = time.time()
        for idx in range(len(self.test_image_loader.LDCT_images)):
            test_zi, test_xi = self.test_image_loader.LDCT_images[idx], self.test_image_loader.NDCT_images[idx]

            whole_G_zi = self.sess.run(self.G_whole_zi,
                                       feed_dict={self.whole_z: test_zi.reshape(self.whole_z.get_shape().as_list())})

            save_file_nm_f = 'from_' + self.test_image_loader.LDCT_image_name[idx]
            save_file_nm_t = 'to_' + self.test_image_loader.NDCT_image_name[idx]
            save_file_nm_g = 'Gen_from_' + self.test_image_loader.LDCT_image_name[idx]

            # 保存一个图像到一个二进制的文件中，格式是.npy
            np.save(os.path.join(npy_save_dir, save_file_nm_f), test_zi)
            np.save(os.path.join(npy_save_dir, save_file_nm_t), test_xi)
            np.save(os.path.join(npy_save_dir, save_file_nm_g), whole_G_zi)

