import torch
import torch.nn as nn
import torch.optim as optim
from environment.GoEnv import Go, TimeStep
from agent.agent import *
from agent.PolicyNetworkAgent import PolicyNetworkAgent
from algorimths.mini_alphago import *
import matplotlib.pyplot as plt
import datetime
import random
from agent.RandomAgent import RandomAgent
from save_model import save_model


def main(iterations=3, policy_iters=2, policy_epochs=100, policy_record_interval=20,  
         value_data_size=8192, value_epochs=100, batch_size=5, lr=1e-4):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f'使用设备: {device}')

    for i in range(iterations):  
        train_policy(num_iterations=policy_iters, num_epochs=policy_epochs, num_record=policy_record_interval,  
                     batch_size=batch_size, lr=lr)  
        print('train_value')
        train_value(num_data=value_data_size, num_epochs=value_epochs, batch_size=batch_size, lr=lr)  

def train_policy(num_iterations=80, num_epochs=500, num_record=20, batch_size=32, lr=1e-4):  
    game_data = []  
    policy_net = PolicyNetwork()  
    opponent_net = PolicyNetwork()  
    opponent = PolicyNetworkAgent(1, opponent_net)  
    policy_player = PolicyNetworkAgent(0, policy_net)  
    agent_pool = [opponent]  

    save_model(policy_net, 0, 'policy_network')  
    
    for iter_num in range(num_iterations):  
        print(f'迭代 = {iter_num} / {num_iterations}')  
        epoch = 0  
        while epoch < num_epochs:  
            data, result = self_play(policy_player, opponent)  
            if result[1] == 1:  
                print(f'获取数据 {epoch} / {num_epochs}')  
                game_data.extend(data)  
                epoch += 1  
        print(f'数据获取完成，总长度: {len(game_data)}')  
        update_policy(game_data, policy_net, 25, batch_size=batch_size, lr=lr, graph=(iter_num + 1) % num_record == 0)  
        
        save_model(policy_net, iter_num, 'policy_network')  
        new_agent = copy.deepcopy(policy_player)  
        agent_pool.append(new_agent)  
        opponent = random.choice(agent_pool)  
        game_data = []  

def save_model(model, iteration, prefix="policy_network"):  
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')  
    filepath = f"{prefix}_iter_{iteration}_{timestamp}.pth"  
    torch.save(model.state_dict(), filepath)  

def self_play(policy_agent, opponent_agent, alternative_agent=None, greedy=True):  
    env = Go()  
    timestep = env.reset()  
    episode_history = []  
    signal_triggered = False  
    record_move = False  
    final_state = ''  
    
    while not timestep.last():  
        current_player = timestep.observations["current_player"]  
        if alternative_agent is None:  
            if current_player == 0:  
                agent_output = policy_agent.step(timestep, greedy=greedy)  
            else:  
                agent_output = opponent_agent.step(timestep, greedy=greedy)  
        else:  
            if current_player == 0:  
                if np.random.rand() < 0.1 and not signal_triggered:  
                    signal_triggered = True  
                    record_move = True  
                if signal_triggered:  
                    agent_output = policy_agent.step(timestep, greedy=greedy)  
                else:  
                    agent_output = alternative_agent.step(timestep, greedy=greedy)  
            else:  
                agent_output = opponent_agent.step(timestep, greedy=greedy)  
        
        action = agent_output.action  
        timestep = env.step(action)  
        episode_history.append((timestep, action))  
        
        if record_move:  
            final_state = episode_history[-1][0].observations["info_state"][0]  
            record_move = False  
    
    reward = timestep.rewards[0]  
    return (episode_history, (final_state, reward))  

def update_policy(play_data, policy_net, epochs, batch_size=32, lr=1e-4, graph=False):  
    def sample_data(data, sample_size=2000):  
        policy_only = [d for d in data if d[0].observations["current_player"] == 0]  
        print(len(data))  
        return random.sample(policy_only, min(sample_size, len(policy_only)))  
    
    sampled_data = sample_data(play_data)  
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)  
    total_losses = []  
    
    for epoch in range(epochs):  
        random.shuffle(sampled_data)  
        print(f'更新策略，第 {epoch} 个周期')  
        epoch_loss = 0  
        for i in range(0, len(sampled_data), batch_size):  
            batch = sampled_data[i:i + batch_size]  
            states, actions, legal_actions = zip(*[  
                (d[0].observations["info_state"][0], d[1], d[0].observations["legal_actions"][0])   
                for d in batch  
            ])  
            
            states_tensor = torch.tensor(states, dtype=torch.float32, requires_grad=True).to(policy_net.device)  
            if len(states_tensor) < batch_size:  
                states_tensor = states_tensor.view(-1, 1, 5, 5)  
            else:  
                states_tensor = states_tensor.view(batch_size, 1, 5, 5)  
            
            actions_tensor = torch.tensor(actions, dtype=torch.int64).to(policy_net.device)  
            action_pools = [torch.tensor(a, dtype=torch.int64).to(policy_net.device) for a in legal_actions]  
            
            target_one_hot = torch.zeros(batch_size, 26).to(policy_net.device)  
            for idx, action in enumerate(actions_tensor):  
                if action < 26:  
                    target_one_hot[idx, action] = 1  
            
            probs = policy_net(states_tensor)  
            log_probs = torch.log(probs + 1e-9)  
            loss = -(target_one_hot * log_probs).sum() / batch_size  
            print(f'损失 = {loss.item()}')  
            
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            epoch_loss += loss.item()  
        
        average_loss = epoch_loss / (len(sampled_data) / batch_size)  
        total_losses.append(average_loss)  
    
    if graph:  
        plt.plot(total_losses)  
        plt.xlabel('周期')  
        plt.ylabel('损失')  
        plt.title('策略网络损失曲线')  
        plt.savefig(f'policy_network_loss_epoch_{epoch}.png')  
        plt.close()  

def train_value(num_data=8192, num_epochs=100, batch_size=64, lr=0.001):  
    value_net = ValueNetwork()  
    policy_net = PolicyNetwork()  
    
    def load_policy_params(net, path):  
        try:  
            net.load_state_dict(torch.load(path))  
            net.eval()  
            print(f"已从 {path} 加载策略网络参数")  
        except Exception as e:  
            print(f"加载策略网络参数失败: {e}")  

    load_policy_params(policy_net, 'policy_network_iter_79.pth')  
    policy_player = PolicyNetworkAgent(0, policy_net)  
    opponent = RandomAgent(1)  
    
    policy_net_backup = PolicyNetwork()  
    load_policy_params(policy_net_backup, 'policy_network_iter_0.pth')  
    alternative_player = PolicyNetworkAgent(0, policy_net_backup)  
    
    dataset = []  
    for n in range(num_data):  
        print(f'获取数据 {n} / {num_data}')  
        result = self_play(policy_player, opponent, alternative_player, greedy=False)[1]  
        if len(result[0]) != 0:  
            dataset.append(result)  
    
    loss_history = []  
    for epoch in range(num_epochs):  
        print(f'训练价值网络，周期 {epoch} / {num_epochs}')  
        update_value_network(dataset, value_net, loss_history, batch_size=batch_size, lr=lr)  
        save_model(value_net, epoch, 'value_network')  
    
    plt.plot(loss_history)  
    plt.xlabel('周期')  
    plt.ylabel('损失')  
    plt.title('价值网络损失曲线')  
    plt.savefig('value_network_loss_curve_epoch.png')  
    plt.close()  

def update_value_network(data, network, loss_hist, batch_size=64, lr=0.001):  
    optimizer = optim.Adam(network.parameters(), lr=lr)  
    total_loss = 0  
    
    for i in range(0, len(data), batch_size):  
        batch = data[i:i + batch_size]  
        states, rewards = zip(*batch)  
        states_tensor = torch.tensor(states, dtype=torch.float32).view(-1, 1, 5, 5).to(network.device)  
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(network.device)  
        
        optimizer.zero_grad()  
        predictions = network(states_tensor).squeeze()  
        loss = torch.mean((predictions - rewards_tensor) ** 2)  
        loss.backward()  
        optimizer.step()  
        
        total_loss += loss.item()  
    
    average_loss = total_loss / (len(data) / batch_size)  
    loss_hist.append(average_loss)  

if __name__ == "__main__":  
    main()