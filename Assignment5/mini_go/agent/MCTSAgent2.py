import random  # 导入随机模块用于随机选择
import numpy as np  # 导入NumPy进行数值运算
import copy  # 导入copy模块以实现深拷贝
import collections  # 导入collections以使用命名元组
from algorimths.mini_alphago import *
StepOutput = collections.namedtuple("step_output", ["action", "probs"])

from agent.agent import Agent  # 从agent模块导入基础Agent类

# 定义一个命名元组来存储每一步的行动和概率
StepResult = collections.namedtuple("step_result", ["action", "probability"])


class MCTSAgent(Agent):
    def __init__(self, _id, fast_policy_network_path,num_simulations=100):
        super().__init__()
        self.player_id = _id
        self.num_simulations = num_simulations
        self.fast_policy_network = FastPolicyNetwork()
        self.fast_policy_network.load_state_dict(torch.load(fast_policy_network_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fast_policy_network.to(self.device)
    def state_to_tensor(self, state):
        state = torch.tensor(state, dtype=torch.float32).view(-1, 1, 5, 5).to(self.device)
        return state
    def step(self, state, env, timestep):
        root = TreeNode(state)
        for _ in range(self.num_simulations):
            env_copy = copy.deepcopy(env)
            timestep_copy = copy.deepcopy(timestep)
            node = root
            search_path = []

            # Selection
            while not node.is_leaf():
                action = random.choice(list(node.children.keys()))
                node = node.children[action]
                search_path.append(node)
            leaf_value = 0
            if timestep.last():
                current_player=timestep.observations['current_player']
                leaf_value = timestep.rewards[current_player]
            else:
                if node.is_leaf():
                    action_probs = self.expand_node(node)
                    current_player=timestep.observations['current_player']
                    valid_actions = timestep.observations['legal_actions'][current_player]
                    action_probs = [action_probs[action] for action in valid_actions]
                    for action in range(len(action_probs)):
                        env_copy_copy = copy.deepcopy(env_copy)
                        timestep_copy = env_copy_copy.step(valid_actions[action])
                        current_player=timestep_copy.observations['current_player']
                        new_state = timestep_copy.observations['info_state'][current_player]
                        new_node = TreeNode(new_state, parent=node)
                        new_node.prior_prob = action_probs[action]
                        node.children[valid_actions[action]] = new_node
                leaf_value = self.simulate(env_copy_copy,timestep_copy)
            self.backpropagate(search_path, leaf_value)
        action_counts = np.array([child.visit_count for child in root.children.values()])
        best_action = list(root.children.keys())[np.argmax(action_counts)]
        return StepOutput(action=best_action, probs=1.0)

    def simulate(self, env, timestep):
        envv=copy.deepcopy(env)
        timestepv=copy.deepcopy(timestep)
        while not timestepv.last():
            valid_actions = timestepv.observations['legal_actions'][timestepv.observations['current_player']]
            current_player=timestepv.observations['current_player']
            state=timestepv.observations['info_state'][current_player]
            state=self.state_to_tensor(state)
            probs=self.fast_policy_network(state).detach().cpu().numpy()[0]
            probs=[probs[index] for index in valid_actions]
            action = valid_actions[np.argmax(probs)]
            timestepv = envv.step(action)
        current_player=timestepv.observations['current_player']
        return timestepv.rewards[current_player]

    def backpropagate(self, search_path, value):
        for node in search_path:
            node.visit_count += 1
            node.total_value += value
    def expand_node(self, node):
        # print(node.state)
        state_tensor = self.state_to_tensor(node.state)
        state_tensor = state_tensor.to(self.device)
        action_probs = self.fast_policy_network(state_tensor).detach().cpu().numpy()[0]
        return action_probs
