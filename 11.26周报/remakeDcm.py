import pydicom
import numpy as np
import glob

def ToDcm(low_list,npy_list):
    for i,path in enumerate(low_list):
        slice = pydicom.dcmread(path)  # 读取dicom文件
        intercept = slice.RescaleIntercept
        slope =slice.RescaleSlope
        # val = dataset.data_element('Columns').value          #根据TAG获得其值,可以读写所有tag
        pixeldata = slice.pixel_array  # 获得图像数据的矩阵形式，只读
        #databyte = slice.PixelData  # 获得图像的byte数据，可直接读写
        #datanew = pixeldata[0:512, 0:512]  # 截取原图像的一部分
        slice.Rows, slice.Columns = pixeldata.shape  # 图像矩阵大小的另一种快速读写方法
        newArray=np.load(npy_list[i])
        newArray=(newArray-intercept)/slope
        data16 = np.int16(newArray)  # 必须转为int16
        # dataset.pixel_array.data = data16               #第一种修改图像数据的方法，直接修改像素值
        slice.PixelData = data16.tobytes()  # 第二种修改图像数据的方法，修改byte值，建议用这种方式
        file='./Pred_Lowct/pred_'+str(i)+'.dcm'
        slice.save_as(file)  # 保存为新dcm文件

lowlist=glob.glob("./dataSet/test/C016/Low Dose Images/"+"*.dcm")
npylist=glob.glob("./predNPY/"+"*.npy")
#print(len(lowlist))
#print(len(npylist))
ToDcm(lowlist,npylist)