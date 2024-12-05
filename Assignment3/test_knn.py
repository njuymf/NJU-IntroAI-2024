import os
import sys
import pygame
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler  # 添加标准缩放器

from play import AliensEnvPygame
from extract_features import extract_features  # 确保导入正确的特征提取函数


def main():
    # 加载模型和标准缩放器
    model_path = os.path.join('logs', 'game_records_lvl4_2024-11-18_08-38-15', 'gameplay_model_knn.pkl')
    scaler_path = os.path.join('logs', 'game_records_lvl4_2024-11-18_08-38-15', 'scaler.pkl')
    with open(model_path, 'rb') as f:
        knn_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
         = pickle.load(f)

    pygame.init()

    env = AliensEnvPygame(level=4, render=False)

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
        # 提取特征
        features = extract_features(observation)

        # 将特征转换为二维数组
        features = np.array(features).reshape(1, -1)

        # 使用相同的标准缩放器进行特征缩放
        features_scaled = scaler.transform(features)

        # 使用缩放后的特征进行预测
        action = knn_model.predict(features_scaled)[0]
        action += 1  # 由于模型预测的是 0, 1, 2，所以需要加 1

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