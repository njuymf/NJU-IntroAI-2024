import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment.GoEnv import Go, TimeStep
import copy

class PolicyNetwork(nn.Module):
    def __init__(self, board_size=5):
        super(PolicyNetwork, self).__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size + 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * board_size * board_size, 128)
        self.fc2 = nn.Linear(128, self.num_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        # print(probs)
        return probs

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).view(-1,1,5,5).to(self.device)   
        output = self.forward(state)
        return np.array(output[0].detach().tolist())


class ValueNetwork(nn.Module):
    def __init__(self, board_size=5):
        super(ValueNetwork, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * board_size * board_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value[0][0]

    def predict(self, state):
        with torch.no_grad():
            return self.forward(state).cpu().numpy()

    def train(self, states, targets):
        # Training code for value network
        pass

class FastPolicyNetwork(nn.Module):
    def __init__(self, board_size=5):
        super(FastPolicyNetwork, self).__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size + 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * board_size * board_size, 128)
        self.fc2 = nn.Linear(128, self.num_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return self.forward(state).numpy()

class MiniAlphaGo:
    def __init__(self, policy_path,value_path,fast_policy_path,board_size=5):
        self.board_size = board_size
        self.policy_network = PolicyNetwork(board_size)
        self.policy_network.load_state_dict(torch.load(policy_path))
        self.value_network = ValueNetwork(board_size)
        self.value_network.load_state_dict(torch.load(value_path))
        self.fast_policy_network = FastPolicyNetwork(board_size)
        self.fast_policy_network.load_state_dict(torch.load(fast_policy_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        self.fast_policy_network.to(self.device)
    def mcts(self, state,env,time_step, num_simulations=100):
        # MCTS implementation using policy_network, value_network, and fast_policy_network
        root = TreeNode(state)

        for _ in range(num_simulations):
            env_copy = copy.deepcopy(env)
            timestep = copy.deepcopy(time_step)
            node = root
            search_path = []

            # choose
            while not node.is_leaf():
                action_probs = self.select_action_probs(node)
                valid_actions = list(node.children.keys())
                action_probs = [action_probs[action] for action in valid_actions]
                action = valid_actions[self.select_action_by_ucb1(node, action_probs)]
                node = node.children[action]
                search_path.append(node)

            leaf_value = 0
            if timestep.last():
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

        key_list = list(root.children.keys())
        action_counts = np.array([child.visit_count for child in list(root.children.values())])
        if 25 in key_list and len(key_list)>1:
            key_list.remove(25)
            action_counts = np.array([root.children[action].visit_count for action in list(root.children.keys()) if action!=25])
        value_probs=[]
        for action in key_list:
            envv = copy.deepcopy(env)
            timestepv = envv.step(action)
            current_player=timestepv.observations['current_player']
            statev=timestepv.observations['info_state'][current_player]
            statev = torch.tensor(statev, dtype=torch.float32).view(1, 1, 5, 5).to(self.device)
            # print('shape',statev.shape)
            value_probs.append(self.value_network.predict(statev))
        value_probs=np.array(value_probs)
        probs = action_counts / np.sum(action_counts)
        probs = np.exp(probs) / np.sum(np.exp(probs))
        # Normalize value_probs to have the same scale as probs
        value_probs = value_probs / np.sum(value_probs)
        # Combine probs and value_probs in a 1:1 ratio
        probs = 0.5 * probs + 0.5 * value_probs
        # print(probs)   
        best_action = key_list[np.argmax(probs)]
        return best_action,probs[np.argmax(probs)]
    def select_action_probs(self, node):
        state_tensor = self.state_to_tensor(node.state)
        state_tensor = state_tensor.to(self.device)
        policy_probs = self.policy_network(state_tensor).detach().cpu().numpy()[0]
        return policy_probs
    def select_action_by_ucb1(self, node, action_probs):
        exploration_param = 1.41
        visit_counts = np.array([child.visit_count for child in list(node.children.values())])
        total_visit_count = node.visit_count
        q_values = np.array([child.total_value / (1 + child.visit_count) for child in list(node.children.values())])
        action_probs = np.array(action_probs, dtype=np.float32)
        # print(len(q_values),len(action_probs))
        ucb_scores = q_values + exploration_param * action_probs * np.sqrt(total_visit_count) / (1 + visit_counts)
        return np.argmax(ucb_scores)
    def expand_node(self, node):
        # print(node.state)
        state_tensor = self.state_to_tensor(node.state)
        state_tensor = state_tensor.to(self.device)
        action_probs = self.policy_network(state_tensor).detach().cpu().numpy()[0]
        return action_probs

    def simulate(self, env,timestep):
        env_copy=copy.deepcopy(env)
        current_player=timestep.observations['current_player']
        current_state = timestep.observations['info_state'][current_player]
        while not timestep.last():
            current_player=timestep.observations['current_player']
            available_actions = timestep.observations['legal_actions'][current_player]
            statev=timestep.observations['info_state'][current_player]
            statev=torch.tensor(statev, dtype=torch.float32).view(-1, 1, 5, 5).to(self.device)
            probs=self.fast_policy_network(statev).detach().cpu().numpy()[0]
            available_probs=[probs[index] for index in available_actions]
            action = available_actions[np.argmax(available_probs)]
            timestep=env_copy.step(action)
        current_player=timestep.observations['current_player']
        return timestep.rewards[current_player]

    def backpropagate(self, search_path, value):
        for node in search_path:
            node.visit_count += 1
            node.total_value += value

    def state_to_tensor(self, state):
        state = torch.tensor(state, dtype=torch.float32).view(-1, 1, 5, 5).to(self.device)
        return state
    def step(self, state,env,time_step):
        # Use MCTS to select the best action
        action,probs = self.mcts(state,env,time_step)
        return action,probs


class TreeNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior_prob = 1.0

    def is_leaf(self):
        return len(list(self.children.keys())) == 0
    
def softmax(x):

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
