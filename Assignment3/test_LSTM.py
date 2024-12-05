import os  
import sys  
import torch  
import pygame  
import numpy as np  
import torch.nn as nn  
from sklearn.preprocessing import StandardScaler  
from play import AliensEnvPygame  
from extract_features import extract_features  
import pickle  # 用于加载保存的标准化器  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import RandomOverSampler  # 改为随机过采样
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from Load_game_records import load_game_records

# 确保模型结构与训练时完全一致  
class ImprovedLSTM(nn.Module):  
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.5):  
        super(ImprovedLSTM, self).__init__()  
        # 添加批归一化层  
        self.batch_norm = nn.BatchNorm1d(input_size)  
        # 双向LSTM  
        self.lstm = nn.LSTM(  
            input_size,  
            hidden_size,  
            num_layers=num_layers,  
            batch_first=True,  
            dropout=dropout,  
            bidirectional=True  
        )  
        # 注意力机制  
        self.attention = nn.MultiheadAttention(  
            embed_dim=hidden_size * 2,  
            num_heads=8,  
            dropout=dropout  
        )  
        # 多层全连接网络  
        self.fc_layers = nn.Sequential(  
            nn.Linear(hidden_size * 2, hidden_size),  
            nn.ReLU(),  
            nn.Dropout(dropout),  
            nn.Linear(hidden_size, hidden_size // 2),  
            nn.ReLU(),  
            nn.Dropout(dropout),  
            nn.Linear(hidden_size // 2, num_classes)  
        )  
        # 残余连接  
        self.residual = nn.Linear(input_size, hidden_size * 2)  
        # 层归一化  
        self.layer_norm = nn.LayerNorm(hidden_size * 2)  

    def forward(self, x):  
        # 批归一化  
        batch_size, seq_len, features = x.size()  
        x = x.reshape(-1, features)  
        x = self.batch_norm(x)  
        x = x.reshape(batch_size, seq_len, features)  
        
        # LSTM处理  
        lstm_out, _ = self.lstm(x)  
        
        # 注意力机制  
        attention_out, _ = self.attention(  
            lstm_out.permute(1, 0, 2),  
            lstm_out.permute(1, 0, 2),  
            lstm_out.permute(1, 0, 2)  
        )  
        attention_out = attention_out.permute(1, 0, 2)  
        
        # 残余连接  
        residual = self.residual(x)  
        combined = attention_out + residual  
        
        # 层归一化  
        normalized = self.layer_norm(combined)  
        
        # 取最后一个时间步  
        out = normalized[:, -1, :]  
        
        # 通过多层全连接网络  
        out = self.fc_layers(out)  
        
        return out  

def main():  
    pygame.init()  

    # 设备选择  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Using device: {device}")  

    env = AliensEnvPygame(level=3, render=False)  

    # 加载模型权重路径  
    model_path = os.path.join('logs', 'game_records_lvl3_2024-11-22_15-05-22', 'improved_gameplay_model_lstm.pth')  
    if not os.path.exists(model_path):  
        print(f"模型文件未找到: {model_path}")  
        return  

    # 加载保存的标准化器  
    scaler_path = os.path.join('logs', 'game_records_lvl3_2024-11-22_14-56-10', 'scaler.pkl')  
    if not os.path.exists(scaler_path):  
        print(f"标准化器文件未找到: {scaler_path}")  
        return  

    with open(scaler_path, 'rb') as f:  
        scaler = pickle.load(f)  

    # 确定输入特征大小  
    input_size = len(extract_features(env.reset()))  
    
    # 加载数据以确定类别数量  
    data = load_game_records()  # 需要确保此函数可用并正确导入  
    if not data:  
        print("未找到可用的游戏记录以确定类别数量")  
        return  
    y = [action for _, action in data]  
    num_classes = len(np.unique(y))  
    print(f"类别数量: {num_classes}")  

    # 初始化模型  
    model = ImprovedLSTM(input_size, num_classes).to(device)  

    # 加载模型权重  
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.eval()  # 设置为评估模式  

    print("模型加载完成")  

    observation = env.reset()  

    grid_image = env.do_render()  

    mode = grid_image.mode  
    size = grid_image.size  
    data_image = grid_image.tobytes()  
    pygame_image = pygame.image.fromstring(data_image, size, mode)  

    screen = pygame.display.set_mode(size)  
    pygame.display.set_caption('Aliens Game - AI Playing')  

    screen.blit(pygame_image, (0, 0))  
    pygame.display.flip()  

    done = False  
    total_score = 0  
    step = 0  
    while not done:  
        # 特征提取和标准化  
        features = extract_features(observation)  
        features_scaled = scaler.transform(features.reshape(1, -1))  # 使用训练时的 scaler  

        # 转换为 PyTorch 张量并调整形状以适应 LSTM 输入  
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(1).to(device)  # 形状为 (1, 1, input_size)  

        # 使用模型预测  
        with torch.no_grad():  
            action_probs = model(features_tensor)  
            action = torch.argmax(action_probs, dim=1).cpu().numpy()[0]  

        observation, reward, game_over, info = env.step(action)  
        total_score += reward  
        print(f"Step: {step}, Action taken: {action}, Reward: {reward}, Done: {game_over}, Info: {info}")  
        step += 1  

        grid_image = env.do_render()  
        mode = grid_image.mode  
        size = grid_image.size  
        data_image = grid_image.tobytes()  
        pygame_image = pygame.image.fromstring(data_image, size, mode)  

        screen.blit(pygame_image, (0, 0))  
        pygame.display.flip()  

        if game_over or step > 500:  
            print("Game Over!")  
            print(f"Info: {info}, Score: {total_score}")  
            done = True  

        pygame.time.delay(100)  

    env.save_gif(filename=f'replay_ai.gif')  

    pygame.quit()  
    sys.exit()  

if __name__ == '__main__':  
    main()