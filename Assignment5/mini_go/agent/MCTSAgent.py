import random  # 导入随机模块用于随机选择
import numpy as np  # 导入NumPy进行数值运算
import copy  # 导入copy模块以实现深拷贝
import collections  # 导入collections以使用命名元组

from agent.agent import Agent  # 从agent模块导入基础Agent类

# 定义一个命名元组来存储每一步的行动和概率
StepResult = collections.namedtuple("step_result", ["action", "probability"])

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # 当前节点表示的游戏状态
        self.parent = parent  # 父节点
        self.children = {}  # 子节点的字典
        self.visits = 0  # 节点被访问的次数
        self.value_sum = 0.0  # 节点累计的总奖励值

    def has_children_nodes(self):
        return bool(self.children)  # 判断是否有子节点

    def add_value(self, value):
        self.visits += 1  # 增加访问次数
        self.value_sum += value  # 累加奖励值

class MCTSAgent(Agent):
    def __init__(self, player_id, simulations=100, exploration=1.4):
        super().__init__()  # 初始化父类
        self.player_id = player_id  # 代理的玩家ID
        self.simulations = simulations  # 模拟次数
        self.exploration = exploration  # UCB的探索常数

    def clone_env_and_timestep(self, env, timestep):
        # 深拷贝环境和时间步，以避免修改原始对象
        return copy.deepcopy(env), copy.deepcopy(timestep)

    def decide_step(self, state, env, timestep):
        root = Node(state)  # 创建MCTS的根节点
        for _ in range(self.simulations):  # 进行多次模拟
            env_copy, timestep_copy = self.clone_env_and_timestep(env, timestep)
            current_node = root  # 从根节点开始
            path = []  # 记录经过的节点路径

            # 选择阶段：遍历树直到找到叶子节点
            while current_node.has_children_nodes():
                action, current_node = self.choose_child(current_node)  # 根据UCB选择子节点
                path.append(current_node)  # 添加到路径中
                timestep_copy = env_copy.step(action)  # 依据动作推进环境

            # 扩展阶段：如果游戏未结束，扩展叶子节点
            if not timestep_copy.last():
                legal_actions = timestep_copy.observations['legal_actions'][timestep_copy.observations['current_player']]
                for act in legal_actions:
                    info_state = timestep_copy.observations['info_state'][timestep_copy.observations['current_player']]
                    child = Node(info_state, parent=current_node)  # 创建子节点
                    current_node.children[act] = child  # 添加到当前节点的子节点中

            # 模拟阶段：如果游戏结束，获取奖励；否则进行随机模拟
            if timestep_copy.last():
                current_player = timestep_copy.observations['current_player']
                reward = timestep_copy.rewards[current_player]  # 获取当前玩家的奖励
            else:
                reward = self.run_simulation(env_copy, timestep_copy)  # 进行模拟获取奖励

            # 回传阶段：更新路径上的节点
            self.update_path(path, reward)

        # 选择访问次数最多的动作作为最佳动作
        action_visit = {act: child.visits for act, child in root.children.items()}
        best_action = max(action_visit, key=action_visit.get)
        return StepResult(action=best_action, probability=1.0)  # 返回最佳动作及其概率

    def compute_ucb(self, parent, child):
        # 计算子节点的UCB得分
        exploitation = child.value_sum / (child.visits + 1e-4)  # 平均价值
        exploration = self.exploration * np.sqrt(np.log(parent.visits + 1) / (child.visits + 1e-4))  # 探索项
        return exploitation + exploration  # UCB得分

    def choose_child(self, node):
        # 选择具有最高UCB得分的子节点
        best_score = -float('inf')
        best_move = None
        best_node = None
        for move, child in node.children.items():
            score = self.compute_ucb(node, child)
            if score > best_score:
                best_score = score
                best_move = move
                best_node = child
        return best_move, best_node  # 返回最佳动作及其子节点

    def run_simulation(self, env, timestep):
        # 从当前状态进行随机模拟直到游戏结束
        env_sim, timestep_sim = self.clone_env_and_timestep(env, timestep)
        while not timestep_sim.last():
            actions = timestep_sim.observations['legal_actions'][timestep_sim.observations['current_player']]
            selected_action = random.choice(actions)  # 随机选择动作
            timestep_sim = env_sim.step(selected_action)  # 执行动作
        current_player = timestep_sim.observations['current_player']
        return timestep_sim.rewards[current_player]  # 返回奖励

    def update_path(self, path, value):
        # 更新搜索路径上的所有节点
        for node in path:
            node.add_value(value)  # 累加奖励值