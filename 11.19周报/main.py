import tensorflow as tf
from model import wganVgg




tfconfig = tf.ConfigProto(allow_soft_placement=True)  # 是配置tf.Session的运算方式
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
model = wganVgg(sess, dcm_path="E:\\program\\PaperCode\\dataSet", test_patient_no=["C012", "C016"],
                checkpoint_dir="./checkpoint", \
                LDCT_path="Low Dose Images", NDCT_path="Full Dose Images", whole_size=512, patch_size=64, img_channel=1,
                img_vmax=3072, img_vmin=-1024, \
                batch_size=32, model='wgan_vgg', phase='train', num_iter=2000, lambda_=10, lambda_1=0.1, alpha=1e-5,
                beta1=0.5, beta2=0.9)
'''
参数说明：
    test_patient_no:应该是指测试病人编号的列表！【也就是说数据集有由若干个病人的CT图像组成，需要选泽编号为(L067,L291)病人的CT作为测试用图！】

'''
# 这里的train/test函数都是自己写的那个！！！
model.train(True, 2000, 4, 10, 200, 'wgan_vgg')
#model.test("/dataSet/test")