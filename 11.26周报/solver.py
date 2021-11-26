import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from networks import RED_CNN
from measure import compute_measure


class Solver(object):
    def __init__(self,data_loader,save_path,num_epochs, \
                 print_iters,decay_iters,save_iters,patch,lr):
        self.data_loader = data_loader #数据

        self.device = torch.device('cuda')

        self.save_path = save_path #保存模型

        self.num_epochs = num_epochs #迭代周期
        self.print_iters = print_iters #打印间隔次数！
        self.decay_iters = decay_iters #学习率衰退的间隔次数，
        self.save_iters = save_iters #保存的间隔次数；

        self.patch = patch #图像的尺寸！

        self.REDCNN = RED_CNN()
        #将读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
        self.REDCNN.to(self.device)

        self.lr = lr #学习率
        self.criterion = nn.MSELoss() #计算均方损失函数！
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        #state_dict()中包含了模型各层和其参数tensor的对应关系。
        torch.save(self.REDCNN.state_dict(), f)

    def lr_decay(self):
        lr = self.lr * 0.5
        #param_groups是优化器的一个参数字典：包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数；
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr #对学习率进行赋值！

    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        for epoch in range(1, self.num_epochs):
            self.REDCNN.train(True)#开始训练模型

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                if self.patch:
                    x = x.view(-1, 1, 64, 64)
                    y = y.view(-1, 1, 64, 64)

                pred = self.REDCNN(x)
                loss = self.criterion(pred, y)#计算预测图像与金标准的均方误差！
                # zero_grad将模型的参数梯度初始化为0
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()

                # 反向传播
                loss.backward()
                # 这个方法会更新模型中所有的参数
                self.optimizer.step()
                #将每次迭代的损失存入数组中！
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    #对学习率进行0.5的递减！
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    #存放的是训练模型
                    self.save_model(total_iters)
                    #存放的是迭代这么多次的损失！
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))


