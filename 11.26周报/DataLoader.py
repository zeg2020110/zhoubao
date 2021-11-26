import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.interpolation import rotate

#自定义数据集的话需要继承这个类是吗？
class ct_dataset(Dataset):
    def __init__(self, saved_path, transform,patch ):

        #获取所有低分辨图像的路径
        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))

        self.transform = transform
        self.patch=patch

        self.input_path = input_path
        self.target_path= target_path

    #数据集的长度
    def __len__(self):
        return len(self.target_path)

    #根据数据集的索引【不过此索引没有什么意义】获取数据：可以逐条取出！
    def __getitem__(self, idx):
        input_img, target_img = self.input_path[idx], self.target_path[idx]
        input_img, target_img = np.load(input_img), np.load(target_img)

        if self.transform:
            input_img,target_img = get_transform(input_img,target_img)

        if self.patch:
            input_patches, target_patches = get_patch(input_img,target_img)
            return (input_patches, target_patches)
        else:
            return (input_img, target_img)

def get_transform(LDCT, NDCT):
    '''
        四种变换操作：旋转45度，垂直。水平翻转，缩放！
    '''
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


def get_patch(full_input_img, full_target_img):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = 64, 64

    # 注意这种切割是真的随缘，每一次都是在一整张图片切除一小块！
    for i in range(h//64):
        #函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)



def get_loader(saved_trainpath=None,saved_testpath=None, transform=True,patch=True ,batch_size=32, num_workers=6):
    train_dataset = ct_dataset(saved_trainpath, transform,patch)
    test_dataset = ct_dataset(saved_testpath, transform, patch)
    traindata_loader = DataLoader(dataset=train_dataset , batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testdata_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return traindata_loader,testdata_loader
