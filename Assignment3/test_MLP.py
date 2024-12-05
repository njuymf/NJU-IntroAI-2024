import os
import sys
import torch
import pygame
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from play import AliensEnvPygame
from extract_features import extract_features
from extract_features import extract_features_plus


# 必须与训练时使用的模型结构完全一致
class ImprovedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


def main():
    pygame.init()

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = AliensEnvPygame(level=3, render=False)

    # 加载模型
    model_path = os.path.join('logs', 'game_records_lvl3_2024-11-21_10-08-12', 'improved_gameplay_model.pth')

    # 准备特征提取的标准化器
    scaler = StandardScaler()

    # 确定输入特征大小 - 这里需要根据实际特征数量调整
    input_size = len(extract_features(env.reset()))
    num_classes = 4  # 假设动作空间是4个动作

    # 初始化模型
    model = ImprovedNN(input_size, num_classes).to(device)

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
        features = extract_features_plus(observation)
        features_scaled = scaler.fit_transform(features.reshape(1, -1))

        # 转换为 PyTorch 张量
        features_tensor = torch.FloatTensor(features_scaled).to(device)

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