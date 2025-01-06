import random  
from environment.GoEnv import Go  
import time  
from agent.MCTSAgent2 import MCTSAgent  
from agent.AlphaGoAgent import MiniAlphaGoAgent  
from algorimths.alpha_go import MiniAlphaGo  
import torch  

def initialize_agents(policy_path, value_path, fast_policy_path):  
    """初始化代理及其各自的策略。"""  
    print('正在初始化代理...')  
    minialphago = MiniAlphaGo(policy_path, value_path, fast_policy_path)  
    minialphago_agent = MiniAlphaGoAgent(1, minialphago)  
    mcts_agent = MCTSAgent(0, fast_policy_path)  
    return [mcts_agent, minialphago_agent]  

def play_game(env, agents):  
    """进行一场游戏并返回结果。"""  
    print('正在进行游戏...')  
    time_step = env.reset()  
    while not time_step.last():  
        player_id = time_step.observations["current_player"]  
        state = time_step.observations['info_state'][player_id]  
        agent_output = agents[player_id].step(state, env, time_step)  
        action_list = agent_output.action  
        time_step = env.step(action_list)  
    print('游戏结束！')  
    print('赢家:', time_step.rewards[0])  
    return 1 if time_step.rewards[0] == -1 else 0  

def main():  
    begin = time.time()  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    print('使用 {} 设备'.format(device))  
    
    # 初始化环境和代理  
    env = Go()  
    policy_path = 'parameters3/policy_network_iter_79.pth'  
    value_path = 'parameters3/value_network_iter_99.pth'  
    fast_policy_path = 'parameters3/policy_network_iter_19.pth'  
    agents = initialize_agents(policy_path, value_path, fast_policy_path)  
    
    num_episodes = 20  
    results = []  

    # 进行多场比赛  
    for ep in range(num_episodes):  
        result = play_game(env, agents)  
        results.append(result)  
        print(f'第 {ep + 1}/{num_episodes} 场游戏胜利: {result}')  

    # 输出结果  
    print('耗时:', time.time() - begin)  
    print('胜率:', sum(results) / num_episodes)  

if __name__ == '__main__':  
    main()