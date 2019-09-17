#-*-coding:utf-8-*-

'''
《深度学习-卷积神经网络从入门到精通》中的lenet5实现（p43 - p44），
但是原书公式的效果极差，所以相对原书做了如下修改：
1. 将sigmoid改换成Relu
2. 增加一个84 -> 10的全连接层
3. 将平均池化换成最大池化
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        
        self.fc1 =  nn.Linear(120, 84)
        self.fc2 =  nn.Linear(84, 10)

    def forward(self, input):
        # 28 * 28 - > 28 * 28 -> 14 * 14
        x = F.max_pool2d(F.relu(self.conv1(input)), 2, stride=2)
        # 14 * 14 -> 10 * 10 - > 5 * 5
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, stride=2)
        
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0),-1)
        
        # 84 * 84 -> 10*10
        x = self.fc2(F.relu(self.fc1(x)))
        
        x = F.softmax(x,dim=1)
        #print(x.size)
        return x
        
    
