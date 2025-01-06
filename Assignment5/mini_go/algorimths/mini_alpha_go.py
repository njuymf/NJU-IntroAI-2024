import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from environment.GoEnv import Go, TimeStep  
import copy  

class ActionPolicyModel(nn.Module):  
    def __init__(self, grid_size=5):  
        super(ActionPolicyModel, self).__init__()  
        self.grid_size = grid_size  
        self.action_dim = grid_size * grid_size + 1  
        self.conv_initial = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv_middle = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.fc_hidden = nn.Linear(64 * grid_size * grid_size, 128)  
        self.fc_output = nn.Linear(128, self.action_dim)  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.to(self.device)  

    def forward(self, x):  
        x = F.relu(self.conv_initial(x))  
        x = F.relu(self.conv_middle(x))  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc_hidden(x))  
        logits = self.fc_output(x)  
        probs = F.softmax(logits, dim=1)  
        return probs  

    def predict(self, state):  
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).view(-1, 1, self.grid_size, self.grid_size).to(self.device)  
        output = self.forward(state_tensor)  
        return output.detach().cpu().numpy()[0]  


class StateValueModel(nn.Module):  
    def __init__(self, grid_size=5):  
        super(StateValueModel, self).__init__()  
        self.grid_size = grid_size  
        self.conv_a = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv_b = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.fc_hidden = nn.Linear(64 * grid_size * grid_size, 128)  
        self.fc_value = nn.Linear(128, 1)  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.to(self.device)  

    def forward(self, x):  
        x = F.relu(self.conv_a(x))  
        x = F.relu(self.conv_b(x))  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc_hidden(x))  
        value = self.fc_value(x)  
        return value.squeeze()  

    def predict(self, state):  
        with torch.no_grad():  
            state_tensor = torch.tensor(state, dtype=torch.float32).view(1, 1, self.grid_size, self.grid_size).to(self.device)  
            return self.forward(state_tensor).cpu().numpy()  

    def train(self, states, targets):  
        # 训练逻辑  
        pass  


class QuickPolicyModel(nn.Module):  
    def __init__(self, grid_size=5):  
        super(QuickPolicyModel, self).__init__()  
        self.grid_size = grid_size  
        self.action_dim = grid_size * grid_size + 1  
        self.conv_start = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv_end = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.fc_layer = nn.Linear(64 * grid_size * grid_size, 128)  
        self.fc_output = nn.Linear(128, self.action_dim)  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.to(self.device)  

    def forward(self, x):  
        x = F.relu(self.conv_start(x))  
        x = F.relu(self.conv_end(x))  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc_layer(x))  
        logits = self.fc_output(x)  
        probs = F.softmax(logits, dim=1)  
        return probs  

    def predict(self, state):  
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  
        with torch.no_grad():  
            return self.forward(state_tensor).cpu().numpy()[0]  


class MiniAlphaGo:  
    def __init__(self, policy_weights, value_weights, quick_policy_weights, grid_size=5):  
        self.grid_size = grid_size  
        self.policy_net = ActionPolicyModel(grid_size)  
        self.policy_net.load_state_dict(torch.load(policy_weights))  
        self.value_net = StateValueModel(grid_size)  
        self.value_net.load_state_dict(torch.load(value_weights))  
        self.quick_policy_net = QuickPolicyModel(grid_size)  
        self.quick_policy_net.load_state_dict(torch.load(quick_policy_weights))  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.policy_net.to(self.device)  
        self.value_net.to(self.device)  
        self.quick_policy_net.to(self.device)  

    def mcts(self, state, env, time_step, num_simulations=100):  
        root_node = TreeNode(state)  

        for _ in range(num_simulations):  
            env_clone = copy.deepcopy(env)  
            time_clone = copy.deepcopy(time_step)  
            current_node = root_node  
            path = []  

            # 选择阶段  
            while not current_node.is_leaf():  
                probs = self.select_action_probs(current_node)  
                available_actions = list(current_node.children.keys())  
                probs = [probs[action] for action in available_actions]  
                chosen_index = self.select_action_by_ucb1(current_node, probs)  
                chosen_action = available_actions[chosen_index]  
                current_node = current_node.children[chosen_action]  
                path.append(current_node)  

            # 展开和模拟阶段  
            leaf_value = 0  
            if time_clone.last():  
                leaf_value = time_clone.rewards[time_clone.observations['current_player']]  
            else:  
                if current_node.is_leaf():  
                    action_probs = self.expand_node(current_node)  
                    player = time_clone.observations['current_player']  
                    legal_moves = time_clone.observations['legal_actions'][player]  
                    action_probs = [action_probs[action] for action in legal_moves]  
                    for action in legal_moves:  
                        env_copy = copy.deepcopy(env_clone)  
                        timestep_copy = env_copy.step(action)  
                        new_player = timestep_copy.observations['current_player']  
                        new_state = timestep_copy.observations['info_state'][new_player]  
                        child_node = TreeNode(new_state, parent=current_node)  
                        child_node.prior_prob = action_probs[action]  
                        current_node.children[action] = child_node  

                leaf_value = self.simulate(env_clone, time_clone)  

            # 回传阶段  
            self.backpropagate(path, leaf_value)  

        # 模拟结束后的选择  
        actions = list(root_node.children.keys())  
        visit_counts = np.array([child.visit_count for child in root_node.children.values()])  
        if 25 in actions and len(actions) > 1:  
            idx = actions.index(25)  
            actions.pop(idx)  
            visit_counts = np.delete(visit_counts, idx)  
        
        value_list = []  
        for action in actions:  
            temp_env = copy.deepcopy(env)  
            temp_step = temp_env.step(action)  
            current_player = temp_step.observations['current_player']  
            state = temp_step.observations['info_state'][current_player]  
            state_tensor = torch.tensor(state, dtype=torch.float32).view(1, 1, self.grid_size, self.grid_size).to(self.device)  
            value = self.value_net.predict(state_tensor)  
            value_list.append(value)  
        
        value_array = np.array(value_list)  
        probabilities = visit_counts / visit_counts.sum()  
        value_normalized = value_array / value_array.sum()  
        combined_probs = 0.5 * probabilities + 0.5 * value_normalized  
        best_action = actions[np.argmax(combined_probs)]  
        return best_action, combined_probs[np.argmax(combined_probs)]  

    def select_action_probs(self, node):  
        state_tensor = self.state_to_tensor(node.state)  
        state_tensor = state_tensor.to(self.device)  
        policy_output = self.policy_net.forward(state_tensor).detach().cpu().numpy()[0]  
        return policy_output  

    def select_action_by_ucb1(self, node, action_probs):  
        exploration_factor = 1.41  
        visit_counts = np.array([child.visit_count for child in node.children.values()])  
        total_visits = node.visit_count  
        q_values = np.array([child.total_value / (1 + child.visit_count) for child in node.children.values()])  
        ucb_scores = q_values + exploration_factor * action_probs * np.sqrt(total_visits) / (1 + visit_counts)  
        return np.argmax(ucb_scores)  

    def expand_node(self, node):  
        state_tensor = self.state_to_tensor(node.state).to(self.device)  
        action_probs = self.policy_net.forward(state_tensor).detach().cpu().numpy()[0]  
        return action_probs  

    def simulate(self, env, time_step):  
        env_copy = copy.deepcopy(env)  
        step_copy = time_step  
        current_player = step_copy.observations['current_player']  
        current_state = step_copy.observations['info_state'][current_player]  
        while not step_copy.last():  
            player = step_copy.observations['current_player']  
            available_actions = step_copy.observations['legal_actions'][player]  
            state = step_copy.observations['info_state'][player]  
            state_tensor = torch.tensor(state, dtype=torch.float32).view(-1, 1, self.grid_size, self.grid_size).to(self.device)  
            probs = self.quick_policy_net.forward(state_tensor).detach().cpu().numpy()[0]  
            action_probs = [probs[action] for action in available_actions]  
            chosen_action = available_actions[np.argmax(action_probs)]  
            step_copy = env_copy.step(chosen_action)  
        final_player = step_copy.observations['current_player']  
        return step_copy.rewards[final_player]  

    def backpropagate(self, path, value):  
        for node in path:  
            node.visit_count += 1  
            node.total_value += value  

    def state_to_tensor(self, state):  
        tensor = torch.tensor(state, dtype=torch.float32).view(-1, 1, self.grid_size, self.grid_size).to(self.device)  
        return tensor  

    def step(self, state, env, time_step):  
        action, probability = self.mcts(state, env, time_step)  
        return action, probability  


class TreeNode:  
    def __init__(self, state, parent=None):  
        self.state = state  
        self.parent = parent  
        self.children = {}  
        self.visit_count = 0  
        self.total_value = 0  
        self.prior_prob = 1.0  

    def is_leaf(self):  
        return len(self.children) == 0  


def softmax_function(x):  
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum(axis=0)