#-*-coding:utf-8-*

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np

# 参考https://github.com/amir-saniyan/ZFNet/blob/master/zfnet.py

class ZFNet(nn.Module):

    def __init__(self):
        super(ZFNet, self).__init__()

        #第一层
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, (7,7), stride=2),  # 227 * 227 * 3 -> 110 * 110* 96
            nn.ReLU(), #原书缺失此步骤
            nn.MaxPool2d((3,3), 2, padding= 1) , # 110 * 110* 96 -> 55 * 55 * 96  
            nn.LocalResponseNorm(5)  #原书缺失此步骤
        )

        #第二层
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256,  (5, 5), stride=2), # 55 * 55 * 96  -> 26 * 26 * 256
            nn.ReLU(), #原书缺失此步骤
            nn.MaxPool2d((3,3), 2, padding= 1), # 26 * 26 * 256 -> 13 * 13 * 256
            nn.LocalResponseNorm(5)  #原书缺失此步骤
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, (3, 3), stride= 1, padding=1),  # 13 * 13 * 256 -> 13 * 13 * 384
            nn.ReLU(), #原书缺失此步骤
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, (3, 3), stride= 1 , padding=1),  # 13 * 13 * 384 -> 13 * 13 * 384
            nn.ReLU(), #原书缺失此步骤
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256,  (3, 3), stride= 1, padding=1), # 13 * 13 * 384 -> 13 * 13 * 256
            nn.ReLU(), #原书缺失此步骤
            nn.MaxPool2d((3, 3), 2) #  13 * 13 * 256 -> 6*6*256
        )

        self.layer6 = nn.Sequential(
            nn.Linear(6*6*256, 4096),
            nn.ReLU(), #原书缺失此步骤
        )

        self.layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(), #原书缺失此步骤
        )

        self.layer8 = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open('../test.JPEG')

    img = img.resize((227, 227),Image.ANTIALIAS).convert('RGB')
    np_data = np.array(img)
    
    net = ZFNet().to(device)
    input = torch.from_numpy(np_data).transpose(2,0).unsqueeze(0).float()

    net(input)