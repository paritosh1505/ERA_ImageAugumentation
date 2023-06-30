import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

torch.manual_seed(42)
    
    
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    
    self.conv_block1 = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
         
        
    self.conv_block2 = nn.Sequential(
            nn.Conv2d(32,128,3,padding=1,dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
           

            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
    self.conv_block3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1,groups=64),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,64,1),
            nn.ReLU(),
        
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64,32,3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
    self.conv_block4 = nn.Sequential(
            nn.Conv2d(32,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,10,4),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            
            nn.AvgPool2d(1))

    self.fc = nn.Linear(1*1*15,10)
    self.dropout = nn.Dropout2d(0.01)
  def forward(self,x):
    x =  self.conv_block1(x)
    x = self.dropout(x)
    x =  self.conv_block2(x)
    x = self.dropout(x)
    x = self.conv_block3(x)
    x = self.dropout(x)
    x = self.conv_block4(x)
    x = x.view(-1,10)
        # x = self.fc(x)

    return F.log_softmax(x,dim=-1)
            

def model_summary(model,input_val):
    return summary(model, input_size=input_val)