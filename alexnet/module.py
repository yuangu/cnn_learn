#-*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=4) 
        self.conv2 = nn.Conv2d(96, 256,(5,5),stride=1 )
        self.conv3 = nn.Conv2d(256, 384,(3,3),stride=1 )
        self.conv4 = nn.Conv2d(384, 384, (3,3),stride=1 )
        self.conv5 = nn.Conv2d(384, 256, (3,3),stride=1 )

        self.fc1 = nn.Linear(256 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        # 第一层 
        x =  F.local_response_norm(F.max_pool2d(F.relu( self.conv1(x)), kernel_size=3,stride=2), 4,alpha=0.001/9.0,beta=0.75)
        # 第二层
        x = F.local_response_norm(F.max_pool2d(F.relu( self.conv2(x)), kernel_size=3,stride=2), 4,alpha=0.001/9.0,beta=0.75)
        # 第三层
        x = F.relu(self.conv3(x))
        # 第四层
        x = F.relu(self.conv4(x))
        # 第五层
        x = F.max_pool2d( F.relu(self.conv5(x)), (3,3) ,stride = 2)
        #数据转化成一维向量
        x = x.view(x.size(0), -1 ) 
        # 第六层
        x = F.dropout(F.relu(self.fc1(x)), p= 0.5)
        # 第七层
        x = F.dropout(F.relu(self.fc2(x)), p= 0.5)
        # 第八层
        x = F.softmax(self.fc3(x))


        print(x.size())
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open('../test.JPEG')

    img = img.resize((227, 227),Image.ANTIALIAS).convert('RGB')
    np_data = np.array(img)
    
    net = AlexNet().to(device)
    input = torch.from_numpy(np_data).transpose(2,0).unsqueeze(0).float()

    net(input)