import cv2
import numpy as np
import matplotlib.pylab as plt

img=cv2.imread('zebra.bmp')

mask=np.zeros(img.shape[:2],np.uint8) #掩模：说白了，就是一个把二值图像矩阵与原图相乘，1的像素点就是分割留下的结果！

bgdModel=np.zeros((1,65),np.float64) #想知道怎么用需要去具体了解算法吗？
fgdModel=np.zeros((1,65),np.float64)

rect=(20,55,520,300) #这个就是代替交互操作的框，要根据分割图片的对象进行计算。

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)#关键算法

mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')

img=img*mask2[:,:,np.newaxis]

plt.subplot(121)
plt.imshow(img)
plt.title("GrabCut")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(cv2.cvtColor(cv2.imread('zebra.bmp'),cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.xticks([])
plt.yticks([])
plt.show()