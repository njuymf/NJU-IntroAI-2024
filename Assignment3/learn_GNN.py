import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from extract_features import extract_features
from play import AliensEnvPygame
from Load_game_records import load_game_records
from Load_game_records import inspect_data


# 定义神经网络模型
class ImprovedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # 增加神经元
        self.bn1 = nn.BatchNorm1d(256)          # 添加 Batch Normalization
        self.fc2 = nn.Linear(256, 128)          # 增加神经元
        self.bn2 = nn.BatchNorm1d(128)          # 添加 Batch Normalization
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # 输出层
        return x


def main():
    # 加载游戏记录
    data = load_game_records()
    inspect_data(data)

    if not data:
        print("未找到可用的游戏记录")
        return

    X = []
    y = []
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 检查标签的范围
    if len(np.unique(y)) < 2:
        raise ValueError("目标值缺乏多样性，必须至少有两个不同的类。")

    min_label = np.min(y)
    max_label = np.max(y)
    num_classes = max_label + 1  # 更新类别数量
    print(f"标签范围: 最小值: {min_label}, 最大值: {max_label}, 类别数量: {num_classes}")

    # 转换为 PyTorch 张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # 创建数据集和数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = X.shape[1]  # 输入特征数量

    model = ImprovedNN(input_size, num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 适用于多分类
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("开始训练模型...")
    for epoch in range(20):  # 尝试更多的训练轮数
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()  # 清零梯度
            outputs = model(batch_X)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        print(f'Epoch [{epoch + 1}/20], Loss: {loss.item():.4f}')

    env = AliensEnvPygame(level=3, render=False)

    # 评估模型 - 打印混淆矩阵
    with torch.no_grad():
        model.eval()
        y_pred = torch.argmax(model(X_tensor), dim=-1).numpy()  # 获取预测的类别
    print("混淆矩阵:")
    print(confusion_matrix(y, y_pred))

    # 保存模型
    model_path = f'{env.log_folder}/improved_gameplay_model.pth'
    torch.save(model.state_dict(), model_path)

    print(f"模型已保存到 {model_path}，训练完成")


if __name__ == '__main__':
    main()