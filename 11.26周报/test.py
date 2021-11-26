import os
import torch
from DataLoader import get_loader
from networks import RED_CNN
from measure import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.backends import cudnn



# 反正则化：为了显示降噪后的图像！
def denormalize_(image):
    image = image * (3072.0 -(-1024.0)) + (-1024.0)
    return image


#截断操作：将图像中元素限制在【-160，240】之间！
def trunc(mat):
    mat[mat <= -160.0] = -160.0
    mat[mat >= 240.0] = 240.0
    return mat




def save_fig(x, y, pred, fig_name, original_result, pred_result): #这里的保存路径是指测试图像的保存路径！
    #将张量变为矩阵
    print(111111111111111111)
    x, y, pred = x.numpy(), y.numpy(), pred.numpy()
    #开一个Figure对象，由3个尺寸（30,10）的小图像组成，即Axes对象；
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    #即ax[0]显示原低剂量图像
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=-160.0, vmax=240.0)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                       original_result[1],
                                                                       original_result[2]), fontsize=20)
    # 即ax[1]显示去噪图像
    ax[1].imshow(pred, cmap=plt.cm.gray, vmin=-160.0, vmax=240.0)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                       pred_result[1],
                                                                       pred_result[2]), fontsize=20)
    # 即ax[2]显示全剂量图像
    ax[2].imshow(y, cmap=plt.cm.gray, vmin=-160.0, vmax=240.0)
    ax[2].set_title('Full-dose', fontsize=30)
    file='./ predCT/result_'+str(fig_name)+'.png'
    f.savefig(file)
    print(222222222222222222220)
    plt.close()

def save_npy(pred,fig_name):
    pred=pred.numpy()
    file = './predNPY/pred_' + str(fig_name) + '.npy'
    np.save(file,pred)


def test(data_loader):
    device=torch.device('cuda')
    REDCNN = RED_CNN().to(device)
    path = "E:\\program\\PaperCode\\save\\REDCNN_12000iter.ckpt"
    REDCNN.load_state_dict(torch.load(path))

    # compute PSNR, SSIM, RMSE
    # 将这些切片的指标累加！
    ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

    #被with torch.no_grad()包住的代码，不用跟踪反向梯度计算，
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            shape_ = x.shape[-1]

            #扩充张量的维度！这里注意从数据集取出的维度是三维的！还有一个批数量！

            x = x.unsqueeze(1).float().to(device)#这里在第二个维度增加维度最好，为通道数！
            y = y.unsqueeze(1).float().to(device)

            #x = x.view(-1, 1, shape_, shape_)
            #y = y.view(-1, 1, shape_, shape_)
            #这里的输入维度为（1,1,512,512），也可以进行计算吗？
            pred = REDCNN(x)

            # denormalize, truncate
            #cpu()是将张量转到CPU上！
            #detach将前操作有关的变量的梯度置为空，即grad属性没有被赋值。
            x = trunc(denormalize_(x.view(shape_, shape_).cpu().detach()))
            y = trunc(denormalize_(y.view(shape_, shape_).cpu().detach()))
            pred = trunc(denormalize_(pred.view(shape_, shape_).cpu().detach()))

            #trunc_max:240; trunc_min:-160
            data_range = 240.0 - (-160.0)

            original_result, pred_result = compute_measure(x, y, pred, data_range)
            ori_psnr_avg += original_result[0]
            ori_ssim_avg += original_result[1]
            ori_rmse_avg += original_result[2]
            pred_psnr_avg += pred_result[0]
            pred_ssim_avg += pred_result[1]
            pred_rmse_avg += pred_result[2]

            # save result figure
            #save_fig(x, y, pred, i, original_result, pred_result)
            save_npy(pred,i)


            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(data_loader),
                                                                                        ori_ssim_avg/len(data_loader),
                                                                                        ori_rmse_avg/len(data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(data_loader),
                                                                                              pred_ssim_avg/len(data_loader),
                                                                                              pred_rmse_avg/len(data_loader)))

if __name__ == "__main__":
    cudnn.benchmark = True
    _,testdata_loader = get_loader(saved_trainpath="E:\\program\\PaperCode\\train_data",
                   saved_testpath="E:\\program\\PaperCode\\test_data", \
                   transform=False, patch=False, batch_size=1, num_workers=6)

    test(testdata_loader)