import os
import argparse
import numpy as np
import pydicom
from tqdm import tqdm
import time


def save_dataset(data_path,save_path):#传入的路径是包含病人编号的上级目录，【自己分为train与test】
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #if 'zip' not in d加上这一句话有什么意思,感觉没啥用啊，去掉也不影响结果，因为想不到"zip"in d的这种情况！
    patients_list = sorted([d for d in os.listdir(data_path)])

    for p_ind, patient in enumerate(patients_list):
        patient_input_path = os.path.join(data_path, patient,
                                          "Low Dose Images")
        patient_target_path = os.path.join(data_path, patient,
                                           "Full Dose Images")

        # 这个循环相当于就是循环两次，分别赋予里面的值
        for path in [patient_input_path, patient_target_path]:
            full_pixels = get_pixels_hu(load_scan(path))
            for pi in tqdm(range(len(full_pixels))):
                if 'Low Dose Images' in path:
                    io = 'input'
                else:
                    io = 'target'
                f = normalize(full_pixels[pi])
                f_name = '{}_{}_{}.npy'.format(patient, pi, io)
                np.save(os.path.join(save_path, f_name), f)
        print('转换完成')


def load_scan(path):
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def normalize(image):
    MIN_B = -1024.0
    MAX_B = 3072.0
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return image

if __name__ == "__main__":
    train_datapath="E:\\program\\PaperCode\\dataSet\\train"
    train_savepath="./train_data"
    save_dataset(train_datapath,train_savepath)
    test_datapath = "E:\\program\\PaperCode\\dataSet\\test"
    test_savepath = "./test_data"
    save_dataset(test_datapath, test_savepath)
