#-*- coding:utf-8-*-

import torch
from torch.utils.data import Dataset, DataLoader

from moudle import LeNet
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size=1 
epoch_num=5 #2-0.8323, 5-0.9545
LR = 0.001

data_train = MNIST('../data/mnist',
                  download=True,
                   transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                      
                       ]))

data_test = MNIST('../data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                   
                      #transforms.Resize((32, 32)),
                      transforms.ToTensor()]))



train_loader = DataLoader( 
  dataset= data_train,  #CustomDataset(), 
  batch_size=batch_size, # 批大小 
 
  shuffle=True, # 是否随机打乱顺序 
  num_workers=8, # 多线程读取数据的线程数 
  ) 

test_loader = DataLoader( 
  dataset= data_test,  #CustomDataset(), 
  batch_size=batch_size, # 批大小 
 
  shuffle=True, # 是否随机打乱顺序 
  num_workers=8, # 多线程读取数据的线程数 
  ) 


net = LeNet().to(device)

opt = torch.optim.SGD(net.parameters(), lr=LR)

loss_function = torch.nn.CrossEntropyLoss()

def train():
  net.train()
  for epoch in range(epoch_num):
      total_loss = 0
      epoch_step = 0
      tic = time.time()

      for batch_image, batch_label in train_loader:
          batch_image = batch_image.to(device)
          batch_label  = batch_label.to(device)

          opt.zero_grad() 
          output = net(batch_image)
          loss = loss_function(output, batch_label) 
          
          total_loss += loss
          epoch_step += 1
          
          loss.backward() 
          opt.step()
        
      toc = time.time()    
      print("one epoch does take approximately " + str((toc - tic)) + " seconds),average loss: " + str(total_loss/epoch_step))

#torch.save(net.state_dict(), "./moudle/moudle")      

def test():
  net.eval()
  total_correct = 0
  for batch_image, batch_label in test_loader:
    batch_image = batch_image.to(device)
    batch_label  = batch_label.to(device)

    output = net(batch_image)
    
    pred = output.detach().max(1)[1]
    total_correct += pred.eq(batch_label.view_as(pred)).sum()

  print("total_correct:", float(total_correct) / len(data_test))



if __name__ == "__main__":
  train()
  test()
     
