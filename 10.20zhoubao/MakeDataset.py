import tensorflow as tf
import os
import cv2
# import matplotlib.pyplot as plt
import numpy as np

def get_files(file_dir):
    image_list = []
    for image_name in os.listdir(file_dir):
        image_name_path = os.path.join(file_dir, image_name)
        image_list.append(image_name_path)
    return image_list

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



# 生成的records文件名
writer = tf.compat.v1.python_io.TFRecordWriter("train.tfrecords")
path_list=get_files("/trainn")
images = []
labels = []
for im in path_list:
	# 读取路径下的图片
	lab = cv2.imread(im, 0)
	lab= modcrop(lab,2)
	h,w = lab.shape
	img=updown(lab)
	for j in range(0, h-33 + 1, 33):  # 在（0，h-33+1)区间内，每隔21取一个，即x为：0,33,66,99,132,
		for k in range(0, w-33 + 1, 33):  # y:0,21,42,
			sub_image = img[j:j + 33, k:k + 33]  # 33*33:相当于提取0到32之间元素，33不算！
			sub_label = lab[j:j + 33, k:k + 33]
			images.append(sub_image)  # 在sub_input_sequence末尾加sub_input中元素 但考虑为空
			labels.append(sub_label)

for i in range(len(images)):
	# 图片转化为二进制格式
	img_raw = images[i].tobytes()
	lab_raw=labels[i].tobytes()
	# example对象对label和image数据进行封装
	example = tf.train.Example(features=tf.train.Features(feature={
		"label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[lab_raw])),
		"img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
	}))

	# 序列化为字符串
	writer.write(example.SerializeToString())

writer.close()

