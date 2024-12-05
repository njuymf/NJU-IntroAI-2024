import os  
import pickle  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader, TensorDataset  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report  
import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import RandomOverSampler  # 改为随机过采样
from sklearn.model_selection import train_test_split


from extract_features import extract_features  
from play import AliensEnvPygame  
from Load_game_records import load_game_records  
from Load_game_records import inspect_data  
from extract_features import extract_features_plus  



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
        # 残差连接  
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
        
        # 残差连接  
        residual = self.residual(x)  
        combined = attention_out + residual  
        
        # Layer Normalization  
        normalized = self.layer_norm(combined)  
        
        # 取最后一个时间步  
        out = normalized[:, -1, :]  
        
        # 通过多层全连接网络  
        out = self.fc_layers(out)  
        
        return out


# 数据预处理
def preprocess_data(data):
    """
    数据预处理，包括：
    - 提取特征和标签；
    - 数据检查、清洗；
    - 特征标准化；
    - 通过随机重复平衡数据（代替 SMOTE）。
    """
    X = []
    y = []

    # 提取特征和标签
    from extract_features import extract_features
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    # 转换为 DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['action'] = y

    print("\n数据集基本信息:")
    print(df.info())
    print(df.describe())

    # 缺失值清理
    if df.isnull().sum().any():
        df = df.dropna()
        print("发现并移除了缺失值")

    # 类别分布检查
    print("原始数据类别分布:")
    print(df['action'].value_counts(normalize=True))

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_names])

    # 随机过采样平衡数据
    print("\n应用随机过采样平衡数据...")
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_scaled, df['action'])
    
    # 定义 env 变量
    env = AliensEnvPygame(level=3, render=False)
    
    # 保存 StandardScaler  
    scaler_path = os.path.join(env.log_folder, 'scaler.pkl')  
    with open(scaler_path, 'wb') as f:  
        pickle.dump(scaler, f)  
    print(f"StandardScaler 已保存到 {scaler_path}")  

    return X_resampled, y_resampled, feature_names

def main():  
    # 加载游戏记录  
    data = load_game_records()  
 
    if not data:  
        print("未找到可用的游戏记录")  
        return  
 
    X = []  
    y = []  
    for observation, action in data:  
        features = extract_features_plus(observation)  
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
 
    # 更新处理输入数据的方式  
    seq_length = 1  # 假设每个样本的时间步长为1  
    X = X.reshape((X.shape[0], seq_length, -1))  # 转换为 (样本数，序列长度, 特征数)  
 
    min_label = np.min(y)  
    max_label = np.max(y)  
    num_classes = max_label + 1  # 更新类别数量  
    print(f"标签范围: 最小值: {min_label}, 最大值: {max_label}, 类别数量: {num_classes}")  
    
    
    # 数据预处理  
    X_resampled, y_resampled, feature_names = preprocess_data(data)  
    if X_resampled is None or y_resampled is None:  
        print("数据预处理失败")  
        return  
    
    # 确保 y_resampled 为 NumPy 数组  
    y_resampled = np.array(y_resampled)  

    # **关键修正**：将 X_resampled 转换为三维张量 (样本数, 1, 特征数)  
    X_resampled = X_resampled.reshape((X_resampled.shape[0], 1, X_resampled.shape[1]))  
    print(f"X_resampled 的形状: {X_resampled.shape}")  # 调试信息  

    # 数据分割  
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)  

    # 转换为 PyTorch 张量  
    X_tensor = torch.FloatTensor(X_train)  
    y_train = np.array(y_train)  # 确保 y_train 为 NumPy 数组  
    y_tensor = torch.LongTensor(y_train)  
 
    # 创建数据集和数据加载器  
    dataset = TensorDataset(X_tensor, y_tensor)  
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  
 
    input_size = X_resampled.shape[2]  # 输入特征数量  
    print(f"Input size: {input_size}")  # 调试信息  

    model = ImprovedLSTM(input_size, num_classes)  
 
    # 检查是否有可用的 GPU  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"使用设备: {device}")  
    model.to(device)  # 将模型移动到 GPU  
 
    # 计算每个类别的权重  
    from collections import Counter  
 
    class_counts = Counter(y_train)  
    total_samples = len(y_train)  
    class_weights = []  
 
    for i in range(num_classes):  
        count = class_counts.get(i, 0)  
        if count == 0:  
            weight = 0.0  
        else:  
            weight = total_samples / (num_classes * count)  
        class_weights.append(weight)  
 
    class_weights = torch.FloatTensor(class_weights).to(device)  
 
    # 定义损失函数和优化器  
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 传入类别权重  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
 
    epochs = 1000  # 尝试更多的训练轮数  
    loss = None  
 
    print("开始训练模型...")  
    for epoch in range(epochs):  
        for batch_X, batch_y in dataloader:  
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # 将数据移动到 GPU  
 
            optimizer.zero_grad()  # 清零梯度  
            outputs = model(batch_X)  # 前向传播  
            loss = criterion(outputs, batch_y)  # 计算损失  
            loss.backward()  # 反向传播  
            optimizer.step()  # 更新参数  
 
        if (epoch + 1) % 100 == 0 or epoch == 0:  
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')  
 
    env = AliensEnvPygame(level=3, render=False)  
 
    # 评估模型 - 打印混淆矩阵  
    with torch.no_grad():  
        model.eval()  
        X_test_tensor = torch.FloatTensor(X_test).to(device)  # 将测试数据转换为张量并移动到 GPU  
        y_test_pred = torch.argmax(model(X_test_tensor), dim=-1).cpu().numpy()  # 获取预测的类别，移回 CPU  
 
    # 计算混淆矩阵  
    cm = confusion_matrix(y_test, y_test_pred)  
 
    # 绘制混淆矩阵热图  
    plt.figure(figsize=(10, 7))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,   
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))  
    plt.title('Confusion Matrix')  
    plt.xlabel('Predicted Label')  
    plt.ylabel('True Label')  
    plt.show()  
 
    # 计算准确率  
    accuracy = accuracy_score(y_test, y_test_pred)  
    print(f"准确率: {accuracy:.2%}")  
 
    # 打印分类报告  
    print("\n分类报告:")  
    print(classification_report(y_test, y_test_pred))  
 
    # 保存模型  
    model_path = f'{env.log_folder}/improved_gameplay_model_lstm.pth'  
    torch.save(model.state_dict(), model_path)  
 
    print(f"模型已保存到 {model_path}，训练完成")  
 
  
 

if __name__ == '__main__':  
    main()