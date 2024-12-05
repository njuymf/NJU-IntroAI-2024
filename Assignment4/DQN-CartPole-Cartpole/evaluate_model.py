import gym
import torch
import numpy as np
import warnings
import os
from matplotlib import pyplot as plt
from agent import DDQNAgent, DQNAgent  # 确保导入正确  


def evaluate_model(model_path, env_name='CartPole-v1', num_episodes=100):
    # 设备初始化  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建环境  
    env = gym.make(env_name)
    # 初始化智能体  
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=input_dim,
        action_dim=output_dim,
        buffer_size=10000,
        seed=1234,
        lr=1e-3,
        device=device
    )
    # 加载模型  
    agent.load_model(model_path)

    # 评估指标  
    total_rewards = []

    # 进行多轮评估  
    for episode in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        done = False
        episode_reward = 0
        while not done:
            # 选择动作时不探索  
            action = agent.act_no_explore(state)
            step_result = env.step(action)

            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result

            episode_reward += reward

        total_rewards.append(episode_reward)

        # 计算评估结果
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    return avg_reward, std_reward


def visualize_model_performance(model_path, env_name='CartPole-v1'):
    # 设备和环境初始化  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name, render_mode='human')
    # 初始化智能体  
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = DDQNAgent(
        state_dim=input_dim,
        action_dim=output_dim,
        seed=1234,
        lr=1e-3,
        device=device
    )
    # 加载模型  
    agent.load_model(model_path)

    # 可视化一个回合  
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state = reset_result[0]
    else:
        state = reset_result

    done = False

    while not done:
        # 渲染环境  
        env.render()

        # 选择动作  
        action = agent.act_no_explore(state)
        step_result = env.step(action)

        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, reward, done, _ = step_result

    env.close()


def find_model_paths(folder_path):
    model_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pth"):
                model_path = os.path.join(root, file)
                model_paths.append(model_path)
    return model_paths


if __name__ == "__main__":
    # 评估所有模型
    model_paths = find_model_paths("models/ddqn_datalog")
    dataset = []

    for model_path in model_paths:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print(f"评估模型：{model_path}")
        avg_reward, std_reward = evaluate_model(model_path)
        dataset.append((model_path, avg_reward, std_reward))
        visualize_model_performance(model_path)

    # 评估结果排序,从小到大

    # 评估结果可视化(柱状图)
    model_names = [os.path.basename(model_path) for model_path, _, _ in dataset]
    avg_rewards = [avg_reward for _, avg_reward, _ in dataset]
    std_rewards = [std_reward for _, _, std_reward in dataset]
    x = np.arange(len(model_names))
    fig, ax = plt.subplots()
    ax.bar(x, avg_rewards, yerr=std_rewards, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Average Reward')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.set_title('Model Performance Evaluation')
    plt.tight_layout()
    plt.show()
    print("评估完成")

    # 评估结果可视化(散点图)
    avg_rewards = [avg_reward for _, avg_reward, _ in dataset]
    std_rewards = [std_reward for _, _, std_reward in dataset]
    plt.scatter(avg_rewards, std_rewards)
    plt.xlabel('Average Reward')
    plt.ylabel('Standard Deviation')
    plt.title('Model Performance Evaluation')
    plt.tight_layout()
    plt.show()
