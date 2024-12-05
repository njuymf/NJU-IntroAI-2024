import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义神经网络模型  
class QNetwork(nn.Module):  
    def __init__(self, state_dim, action_dim, seed, fc_units=128, device="cpu"):  
        super(QNetwork, self).__init__()  
        self.seed = torch.manual_seed(seed)  
        self.fc1 = nn.Linear(state_dim, fc_units)  # 第一层全连接层  
        self.fc2 = nn.Linear(fc_units, fc_units)  # 第二层全连接层  
        self.fc3 = nn.Linear(fc_units, action_dim)  # 输出层  
        self.to(device)  

    def forward(self, state):  
        """  
        前向传播  
        """  
        x = F.relu(self.fc1(state))  # 第一层激活  
        x = F.relu(self.fc2(x))      # 第二层激活  
        return self.fc3(x)            # 输出动作值  

