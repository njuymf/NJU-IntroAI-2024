import gym  
from collections import deque  
import random  
import argparse  
import torch  
import numpy as np  
import os  
from datetime import datetime  
from itertools import product  # 用于生成超参数组合  
from export import create_csv_file, append_to_csv, generate_filename  
from visualization import visualize  
from agent import DQNAgent, DDQNAgent  


def get_args():  
    """  
    解析命令行参数  
    """  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--agent_name", type=str, default="ddqn", help="选择智能体类型: dqn 或 ddqn")  
    parser.add_argument("--num_episodes", type=int, default=300, help="训练的总回合数")  
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="每个回合的最大步数")  
    parser.add_argument("--epsilon_start", type=float, default=0.98, help="初始探索率")  
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="最低探索率")  
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.985, help="探索率衰减率")  
    parser.add_argument("--gamma", type=float, default=0.85, help="折扣因子")  
    parser.add_argument("--lr", type=float, default=6e-5, help="学习率")  
    parser.add_argument("--buffer_size", type=int, default=64000, help="经验回放缓冲区大小")  
    parser.add_argument("--batch_size", type=int, default=64, help="每批次的样本大小")  
    parser.add_argument("--update_frequency", type=int, default=1, help="网络更新频率")  
    args = parser.parse_args()  
    return args  


def eval_policy(agent, env):  
    """  
    评估智能体的策略  
    """  
    reset_result = env.reset()  
    if isinstance(reset_result, tuple):  
        state = reset_result[0]  
    else:  
        state = reset_result  
    done = False  
    total_return = 0  
    while not done:  
        action = agent.act_no_explore(state)  
        step_result = env.step(action)  
        if len(step_result) == 5:  
            next_state, reward, terminated, truncated, _ = step_result  
            done = terminated or truncated  
        else:  
            next_state, reward, done, _ = step_result  
        if isinstance(next_state, tuple):  
            next_state = next_state[0]  
        state = next_state  
        total_return += reward  
    return total_return  


def train(args, agent, buffer, env):  
    # 创建模型保存目录  
    os.makedirs('models', exist_ok=True)  

    # 生成带超参数的文件名  
    csv_filename = generate_filename(args)  
    create_csv_file(csv_filename)  

    best_eval_return = float('-inf')  
    for episode in range(args.num_episodes):  
        reset_result = env.reset()  
        if isinstance(reset_result, tuple):  
            state = reset_result[0]  
        else:  
            state = reset_result  
        
        # 动态设定探索率  
        epsilon = max(args.epsilon_end, args.epsilon_start * (args.epsilon_decay_rate ** episode))  

        losses = []  
        total_return = 0  
        for step in range(args.max_steps_per_episode):  
            # 选择并执行动作  
            action = agent.act(state, epsilon)  
            step_result = env.step(action)  
            if len(step_result) == 5:  
                next_state, reward, terminated, truncated, _ = step_result  
                done = terminated or truncated  
            else:  
                next_state, reward, done, _ = step_result  
            if isinstance(next_state, tuple):  
                next_state = next_state[0]  

            # 存储经验到缓冲区  
            buffer.append((state, action, reward, next_state, done))  

            # 如果缓冲区中经验足够，则进行学习  
            if len(buffer) >= args.batch_size:  
                batch = random.sample(buffer, args.batch_size)  
                loss = agent.learn(batch, args.gamma)  
                losses.append(loss.item())  

            total_return += reward  
            state = next_state  

            if done:  
                break  

        # 计算平均损失  
        average_loss = np.mean(losses) if losses else 0.0  
        # 评估策略的回报  
        eval_return = eval_policy(agent, env)  

        # 追加到CSV  
        append_to_csv(csv_filename, episode + 1, step + 1, average_loss, eval_return)  

        print(  
           f"回合 {episode + 1}/{args.num_episodes} 步数 {step + 1}：训练损失 {average_loss:.4f}, 评估回报 {eval_return}")  

    # 可视化训练结果  
    # visualize(csv_filename)  

    # 添加最终模型保存路径  
    final_model_path = f'models/{args.agent_name}_{csv_filename}{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'  
    agent.save_model(final_model_path)  
    print(f"训练完成，最终模型已保存到 {final_model_path}")  


def grid_search_train(hyperparameter_space, args, env, buffer, input_dim, output_dim, device):  
    """网格调参函数"""  
    best_hyperparams = None  
    best_eval_return = float('-inf')  
    results = []  

    # 遍历所有超参数组合  
    for hyperparams in product(*hyperparameter_space.values()):  
        # 更新args参数  
        for i, key in enumerate(hyperparameter_space.keys()):  
            setattr(args, key, hyperparams[i])  

        print(f"正在训练超参数组合: {dict(zip(hyperparameter_space.keys(), hyperparams))}")  

        # 初始化智能体  
        if args.agent_name.lower() == "dqn":  
            agent = DQNAgent(state_dim=input_dim, action_dim=output_dim, buffer_size=args.buffer_size, seed=1234,  
                             lr=args.lr, device=device)  
        elif args.agent_name.lower() == "ddqn":  
            agent = DDQNAgent(state_dim=input_dim, action_dim=output_dim, seed=1234, lr=args.lr, device=device)  
        else:  
            raise ValueError("不支持的智能体类型！请选择 'dqn' 或 'ddqn'。")  

        # 清空缓冲区  
        buffer.clear()  

        # 训练  
        train(args, agent, buffer, env)  

        # 评估当前超参数组合  
        eval_return = eval_policy(agent, env)  
        results.append((dict(zip(hyperparameter_space.keys(), hyperparams)), eval_return))  

        print(f"超参数组合: {dict(zip(hyperparameter_space.keys(), hyperparams))}, 评估回报: {eval_return}")  

        # 保存最佳超参数  
        if eval_return > best_eval_return:  
            best_eval_return = eval_return  
            best_hyperparams = dict(zip(hyperparameter_space.keys(), hyperparams))  

    print(f"最佳超参数组合: {best_hyperparams}, 最佳评估回报: {best_eval_return}")  
    return best_hyperparams, results  


if __name__ == "__main__":  
    print("开始网格调参...")  
    # 使用CUDA加速  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    print("使用设备：", device)  
    args = get_args()  

    # 设置环境  
    env = gym.make("CartPole-v1")  
    buffer = deque(maxlen=args.buffer_size)  

    # 获取输入和输出维度  
    input_dim = env.observation_space.shape[0]  
    output_dim = env.action_space.n  

    # 定义超参数搜索空间  
    hyperparameter_space = {  
        # "lr": [1e-3,9e-4,8e-4,7e-4,6e-4,5e-4,4e-4,3e-4,2e-4, 1e-4,9e-5,8e-5,7e-5,6e-5,5e-5,4e-5,3e-5,2e-5,1e-5,1e-6],  
        # "num_episodes": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        # "gamma": [0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.955, 0.95, 0.945, 0.94, 0.935, 0.93, 0.925, 0.92, 0.915, 0.91, 0.905, 0.9,
        #           0.895, 0.89, 0.885, 0.88, 0.875, 0.87, 0.865, 0.86, 0.855, 0.85, 0.845, 0.84, 0.835, 0.83, 0.825, 0.82, 0.815, 0.81, 0.805, 0.8,
        #           0.795, 0.79, 0.785, 0.78, 0.775, 0.77, 0.765, 0.76, 0.755, 0.75, 0.745, 0.74, 0.735, 0.73, 0.725, 0.72, 0.715, 0.71, 0.705, 0.7,
        #           0.695, 0.69, 0.685, 0.68, 0.675, 0.67, 0.665, 0.66, 0.655, 0.65, 0.645, 0.64, 0.635, 0.63, 0.625, 0.62, 0.615, 0.61, 0.605, 0.6,
        #           0.595, 0.59, 0.585, 0.58, 0.575, 0.57, 0.565, 0.56, 0.555, 0.55, 0.545, 0.54, 0.535, 0.53, 0.525, 0.52, 0.515, 0.51, 0.505, 0.5],  
        # "update_frequency": [1,2, 4, 8, 16, 32, 64, 128,256],  
        # "buffer_size": [1000, 2000, 4000, 8000, 16000, 32000, 64000],  
        # "max_steps_per_episode": [500, 1000, 2000],
        # "batch_size": [4, 8, 16, 32, 64, 128, 256, 512, 1024],  
        # "epsilon_decay_rate": [0.995, 0.990, 0.985, 0.980, 0.975, 0.970, 0.965, 0.960, 0.955, 0.950, 0.945, 0.940, 0.935, 0.930, 0.925, 0.920, 0.915,
        #                        0.910, 0.905, 0.900]  
        # "epsilon_end": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10],
        # "epsilon_start": [1,0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.9]
        
    }  
    

    # 开始网格调参  
    best_hyperparams, all_results = grid_search_train(hyperparameter_space, args, env, buffer, input_dim, output_dim, device)  

    # 输出所有结果和最佳参数  
    for params, eval_return in all_results:  
        print(f"超参数: {params}, 评估回报: {eval_return}")  

    print(f"最佳超参数: {best_hyperparams}")