import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from q_network import QNetwork

# 定义DQN智能体  
class DQNAgent:  
    def __init__(self, state_dim, action_dim, buffer_size, seed, lr, device="cpu"):  
        self.state_dim = state_dim  
        self.action_dim = action_dim  
        self.seed = random.seed(seed)  
        self.device = device  

        # 初始化本地网络和目标网络  
        self.qnetwork_local = QNetwork(state_dim, action_dim, seed).to(device)  
        self.qnetwork_target = QNetwork(state_dim, action_dim, seed).to(device)  
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)  

        self.t_step = 0  

    def act(self, state, eps=0.):  
        """  
        根据当前状态选择动作，使用ε-贪心策略  
        """  
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  

        self.qnetwork_local.eval()  
        with torch.no_grad():  
            action_values = self.qnetwork_local(state_tensor)  
        self.qnetwork_local.train()  

        # 决定是探索还是利用  
        if np.random.random() > eps:  
            return action_values.argmax(dim=1).item()  
        else:  
            return np.random.randint(self.action_dim)  

    def act_no_explore(self, state):  
        """  
        不进行探索，直接选择最佳动作  
        """  
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  

        self.qnetwork_local.eval()  
        with torch.no_grad():  
            action_values = self.qnetwork_local(state_tensor)  
        self.qnetwork_local.train()  

        return action_values.argmax(dim=1).item()  

    def learn(self, experiences, gamma):  
        """  
        从经验中学习，更新网络参数  
        """  
        device = self.device  
        states, actions, rewards, next_states, dones = zip(*experiences)  
        states = torch.from_numpy(np.vstack(states)).float().to(device)  
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)  
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)  
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)  
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)  

        # 计算下一个状态的最大Q值  
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)  
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))  

        # 计算当前状态的Q值  
        Q_expected = self.qnetwork_local(states).gather(1, actions)  

        # 计算损失并反向传播  
        loss = F.mse_loss(Q_expected, Q_targets)  
        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step()  

        # 软更新目标网络参数  
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)  
        return loss  

    def soft_update(self, local_model, target_model, tau):  
        """  
        软更新目标网络参数  
        """  
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):  
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)  
    
    
    def save_model(self, filepath):  
        """  
        保存模型的权重和优化器状态  
        """  
        torch.save({  
            'local_network_state_dict': self.qnetwork_local.state_dict(),  
            'target_network_state_dict': self.qnetwork_target.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict()  
        }, filepath)  

    def load_model(self, filepath):  
        """  
        加载模型的权重和优化器状态  
        """  
        checkpoint = torch.load(filepath)  
        self.qnetwork_local.load_state_dict(checkpoint['local_network_state_dict'])  
        self.qnetwork_target.load_state_dict(checkpoint['target_network_state_dict'])  
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  


# 定义DDQN智能体，继承自DQNAgent  
class DDQNAgent(DQNAgent):  
    def __init__(self, state_dim, action_dim, seed, lr, device="cpu"):  
        super().__init__(state_dim, action_dim, buffer_size=10000, seed=seed, lr=lr, device=device)  

    def learn(self, experiences, gamma):  
        """  
        双重DQN的学习方法，减少过估计偏差  
        """  
        device = self.device  
        states, actions, rewards, next_states, dones = zip(*experiences)  
        states = torch.from_numpy(np.vstack(states)).float().to(device)  
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)  
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)  
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)  
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)  

        # 使用本地网络选择动作  
        action_max = torch.argmax(self.qnetwork_local(next_states), dim=1).unsqueeze(1)  
        Q_targets = rewards + (  
                    gamma * torch.gather(self.qnetwork_target(next_states).detach(), 1, action_max) * (1 - dones))  

        # 计算当前状态的Q值  
        Q_expected = self.qnetwork_local(states).gather(1, actions)  

        # 计算损失并反向传播  
        loss = F.mse_loss(Q_expected, Q_targets)  
        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step()  

        # 软更新目标网络参数  
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)  
        return loss  

