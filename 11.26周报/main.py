import os
import argparse
from torch.backends import cudnn
from DataLoader import get_loader
from solver import Solver


def main(save_path,num_epochs,print_iters,decay_iters,save_iters,patch,lr):
    #对模型里的卷积层进行预先的优化，在每一个卷积层中测试cuDNN提供的所有卷积实现算法，然后选择最快的那个！
    cudnn.benchmark = True

    # 这是保存模型的路径！
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    traindata_loader,_ = get_loader(saved_trainpath="E:\\program\\PaperCode\\train_data",saved_testpath="E:\\program\\PaperCode\\test_data",\
                             transform=True,patch=True ,batch_size=8, num_workers=6)

    solver = Solver(traindata_loader,save_path,num_epochs,\
                 print_iters,decay_iters,save_iters,patch,lr)
    solver.train()



if __name__ == "__main__":
    main(save_path='./save/',num_epochs=100,print_iters=10,\
         decay_iters=500,save_iters=300,patch=True,lr=1e-5)

    '''
    #可以直接运行，直接使用后面中的默认值输入程序！
    parser = argparse.ArgumentParser()
   #parser.add_argument('--mode', type=str, default='train')
    #parser.add_argument('--load_mode', type=int, default=0)
    #parser.add_argument('--data_path', type=str, default='./AAPM-Mayo-CT-Challenge/')
    parser.add_argument('--saved_path', type=str, default='./npy_img/')
    parser.add_argument('--save_path', type=str, default='./save/')
    #parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=int, default=1000)

    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    '''


