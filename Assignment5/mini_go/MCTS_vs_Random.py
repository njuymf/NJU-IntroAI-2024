import random  
from environment.GoEnv import Go  
import time  
from agent.RandomAgent import RandomAgent  
from agent.MonteCarloAgent import MonteCarloTreeSearch  
from agent.MCTSAgent import MCTSAgent

if __name__ == '__main__':  
    begin = time.time()  
    env = Go()  
    agents = [RandomAgent(0), MonteCarloTreeSearch(1)]  # 一个是RandomAgent，一个是MCTSAgent  
    agents = [RandomAgent(0), MCTSAgent(1)]  # 一个是RandomAgent，一个是MCTSAgent
    results = []

    for ep in range(100):  
        time_step = env.reset()  
        while not time_step.last():  
            player_id = time_step.observations["current_player"]  
            if player_id == 0:  
                # RandomAgent 的 step 方法只需要 time_step  
                agent_output = agents[player_id].step(time_step)  
            else:  
                # MCTSAgent 的 step 方法需要 state、env 和 time_step  
                state = time_step.observations["info_state"][player_id]  
                agent_output = agents[player_id].decide_step(state, env, time_step)  
            action_list = agent_output.action  
            time_step = env.step(action_list)  
        print(time_step.observations["info_state"][0])
        print(time_step.rewards[0])  
        results.append(time_step.rewards[0])
        
    print('Results:', results)
    print('Win rate:', results.count(1)/len(results))
        

    print('Time elapsed:', time.time()-begin)