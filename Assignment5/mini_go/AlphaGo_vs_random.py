import random  
from environment.GoEnv import Go  
import time  
from agent.MCTSAgent import MCTSAgent
from agent.AlphaGoAgent import MiniAlphaGoAgent
from algorimths.mini_alphago import MiniAlphaGo
import torch  

def initialize_agents(policy_path, value_path, fast_policy_path):  
    """Initialize the agents with their respective policies."""  
    print('Initializing agents...')
    minialphago = MiniAlphaGo(policy_path, value_path, fast_policy_path)  
    minialphago_agent = MiniAlphaGoAgent(1, minialphago)  
    mcts_agent = MCTSAgent(0, fast_policy_path)  
    return [mcts_agent, minialphago_agent]  

def play_game(env, agents):  
    """Play a single game and return the result."""  
    print('Playing game...')
    time_step = env.reset()  
    while not time_step.last():  
        player_id = time_step.observations["current_player"]  
        state = time_step.observations['info_state'][player_id]  
        agent_output = agents[player_id].step(state, env, time_step)  
        action_list = agent_output.action  
        time_step = env.step(action_list)  
        # print(time_step.observations["info_state"][0])
    print('Game over!')
    print('Winner:', time_step.rewards[0])
    return 1 if time_step.rewards[0] == -1 else 0  

def main():  
    begin = time.time()  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    # Initialize environment and agents  
    env = Go()  
    policy_path = 'parameters2/policy_network_iter_19.pth'  
    value_path = 'parameters2/value_network_iter_19.pth'  
    fast_policy_path = 'parameters2/policy_network_iter_19.pth'  
    agents = initialize_agents(policy_path, value_path, fast_policy_path)  
    
    num_episodes = 20  
    results = []  

    # Play multiple episodes  
    for ep in range(num_episodes):  
        result = play_game(env, agents)  
        results.append(result)  
        print(f'Game {ep + 1}/{num_episodes} is win: {result}')  

    # Output results  
    print('Time elapsed:', time.time() - begin)  
    print('Win rate:', sum(results) / num_episodes)  

if __name__ == '__main__':  
    main()