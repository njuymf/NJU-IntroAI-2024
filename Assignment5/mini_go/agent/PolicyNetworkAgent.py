import random  # 导入随机模块用于随机选择
import numpy as np  # 导入NumPy进行数值运算
import copy  # 导入copy模块以实现深拷贝
import collections  # 导入collections以使用命名元组

from agent.agent import Agent  # 从agent模块导入基础Agent类

StepResult = collections.namedtuple("step_result", ["action", "probability"])

StepOutput = collections.namedtuple("step_output", ["action", "probs"])


class PolicyNetworkAgent(Agent):
    def __init__(self, _id, policy_network):
        super().__init__()
        self.player_id = _id
        self.policy_network = policy_network

    def step(self, timestep,greedy=True):
        epsilon = 0.1  # Epsilon value for epsilon-greedy strategy
        state = timestep.observations["info_state"][0]
        actions_pool = timestep.observations["legal_actions"][0]
        action_probs = self.policy_network.predict(state)
        legal_action_probs = [action_probs[i] for i in actions_pool]
        if not greedy:
            chosen_action = actions_pool[np.argmax(legal_action_probs)]
        else:
            if np.random.rand() < epsilon:
                # Explore: choose a random action
                chosen_action = random.choice(actions_pool)
            else:
            # Exploit: choose the best action
                chosen_action = actions_pool[np.argmax(legal_action_probs)]
        
        return StepOutput(action=chosen_action, probs=legal_action_probs)


