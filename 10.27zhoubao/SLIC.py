from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pylab as plt

img =io.imread("zebra.bmp")

segments=slic(img,n_segments=60,compactness=10) #直接调用函数，可以学习源码！

out=mark_boundaries(img,segments)#将分割的结果进行标记边界！
io.imshow(out)
'''
plt.subplot(121)
plt.title("n_segments=60 compatchness=10")
plt.imshow(out)

'''
plt.show()