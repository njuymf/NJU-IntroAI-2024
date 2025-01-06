
import numpy as np
from operator import itemgetter
import copy
import random


class Node:
    def __init__(self, parent=None, prior_prob=0.0):
        """
        初始化树节点。

        :param parent: 父节点，根节点的父节点为 None。
        :param prior_prob: 当前节点的先验概率，表示该节点被选择的可能性。
        """
        self.parent = parent  # 父节点
        self.children = {}  # 子节点字典，键为动作，值为对应的子节点
        self.visit_count = 0  # 访问次数
        self.value_sum = 0.0  # 节点的价值总和
        self.u_score = prior_prob  # 节点的 U 值（用于探索）
        self.prior = prior_prob  # 节点的先验概率

    def add_children(self, action_probabilities):
        """
        扩展当前节点，添加子节点。

        :param action_probabilities: 动作及其对应的概率列表。
        """
        for action, prob in action_probabilities:
            if action not in self.children:
                self.children[action] = Node(parent=self, prior_prob=prob)

    def select_child(self):
        """
        选择当前节点的最佳子节点，使用 UCT（上置信界）策略。

        :return: 选择的动作及对应的子节点。
        """
        return max(self.children.items(), key=lambda item: item[1].get_score())

    def update(self, reward, c_puct):
        """
        更新当前节点的访问次数和价值。

        :param reward: 当前节点的叶子价值。
        :param c_puct: 探索常数，用于调整 U 值的影响。
        """
        self.visit_count += 1  # 增加访问次数
        self.value_sum += (reward - self.value_sum) / self.visit_count  # 更新价值
        if not self.is_root():
            # 更新 U 值，考虑父节点的访问次数
            self.u_score = c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)

    def propagate(self, reward, c_puct):
        """
        递归更新当前节点及其父节点的价值。

        :param reward: 当前节点的叶子价值。
        :param c_puct: 探索常数。
        """
        if self.parent:
            self.parent.propagate(reward, c_puct)  # 递归更新父节点
        self.update(reward, c_puct)  # 更新当前节点

    def get_score(self):
        """
        计算当前节点的总分数（价值 + U 值）。

        :return: 当前节点的总分数。
        """
        return self.value_sum + self.u_score

    def is_leaf(self):
        """
        检查当前节点是否为叶子节点（即没有子节点）。

        :return: 如果是叶子节点返回 True，否则返回 False。
        """
        return len(self.children) == 0

    def is_root(self):
        """
        检查当前节点是否为根节点。

        :return: 如果是根节点返回 True，否则返回 False。
        """
        return self.parent is None


class MonteCarloTreeSearch:
    def __init__(self, value_func, policy_func=None, rollout_func=None, lambda_param=0.5, c_puct=5,
                 rollout_limit=100, play_depth=10, num_playouts=100):
        """
        初始化蒙特卡洛树搜索。

        :param value_func: 价值函数，用于评估状态的价值。
        :param policy_func: 策略函数，用于获取动作的概率分布。默认为随机策略。
        :param rollout_func: 回合策略函数，用于模拟游戏直到结束。默认为随机策略。
        :param lambda_param: 权重参数，控制价值和回合评估的贡献。
        :param c_puct: 探索常数，影响 U 值的计算。
        :param rollout_limit: 回合模拟的最大步数。
        :param play_depth: 每次模拟的最大深度。
        :param num_playouts: 每次决策时进行的模拟次数。
        """
        self.root = Node()  # 初始化根节点
        self.value_func = value_func  # 价值函数
        self.policy_func = policy_func if policy_func is not None else self.default_policy  # 策略函数
        self.rollout_func = rollout_func if rollout_func is not None else self.default_rollout  # 回合策略函数
        self.lambda_param = lambda_param  # 权重参数
        self.c_puct = c_puct  # 探索常数
        self.rollout_limit = rollout_limit  # 回合模拟的最大步数
        self.play_depth = play_depth  # 每次模拟的最大深度
        self.num_playouts = num_playouts  # 每次决策时进行的模拟次数
        self.current_player = 0  # 当前玩家标识

    def default_policy(self, state, player_id):
        """
        默认的策略函数，随机选择动作并赋予相等的概率。

        :param state: 当前游戏状态。
        :param player_id: 当前玩家 ID。
        :return: 动作及其对应的概率列表。
        """
        available_actions = state.get_available_actions(player_id)
        if not available_actions:
            return []
        prob = 1.0 / len(available_actions)
        return [(action, prob) for action in available_actions]

    def default_rollout(self, state, player_id):
        """
        默认的回合策略函数，随机选择动作并赋予相等的概率。

        :param state: 当前游戏状态。
        :param player_id: 当前玩家 ID。
        :return: 动作及其对应的概率列表。
        """
        available_actions = state.get_available_actions(player_id)
        if not available_actions:
            return []
        prob = 1.0 / len(available_actions)
        return [(action, prob) for action in available_actions]

    def switch_player(self):
        """
        切换当前玩家。
        """
        self.current_player = 1 - self.current_player  # 切换玩家（0 <-> 1）

    def perform_playout(self, state, env, depth):
        """
        执行一次模拟，直到达到指定深度或游戏结束。

        :param state: 当前游戏状态。
        :param env: 游戏环境。
        :param depth: 模拟的最大深度。
        """
        node = self.root  # 从根节点开始
        current_state = state  # 当前状态

        for _ in range(depth):
            if node.is_leaf():
                action_probs = self.policy_func(current_state, self.current_player)  # 获取动作概率
                if not action_probs:
                    break  # 如果没有可用动作，结束模拟
                node.add_children(action_probs)  # 扩展节点
            action, node = node.select_child()  # 选择最佳子节点
            current_state = env.step(action)  # 执行动作
            self.switch_player()  # 切换玩家

        # 评估叶子节点的价值
        if self.lambda_param < 1:
            v = self.value_func(current_state, self.current_player)  # 价值评估
        else:
            v = 0
        if self.lambda_param > 0:
            z = self.rollout_evaluation(current_state, env, self.rollout_limit)  # 回合评估
        else:
            z = 0
        leaf_value = (1 - self.lambda_param) * v + self.lambda_param * z  # 计算叶子价值

        node.propagate(leaf_value, self.c_puct)  # 更新节点

    def rollout_evaluation(self, state, env, limit):
        """
        使用回合策略进行模拟，直到游戏结束。

        :param state: 当前游戏状态。
        :param env: 游戏环境。
        :param limit: 最大模拟步数。
        :return: 当前玩家的最终奖励。
        """
        current_state = state  # 当前状态
        for _ in range(limit):
            actions = self.rollout_func(current_state, self.current_player)  # 获取可用动作
            if not actions:
                break  # 如果没有可用动作，结束模拟
            action = random.choice(actions)[0]  # 随机选择一个动作
            current_state = env.step(action)  # 执行动作
            self.switch_player()  # 切换玩家
            if current_state.last():  # 检查游戏是否结束
                break
        return current_state.rewards[0]  # 返回当前玩家的奖励

    def choose_action(self, state, env):
        """
        根据模拟结果选择最佳动作。

        :param state: 当前游戏状态。
        :param env: 游戏环境。
        :return: 选择的最佳动作。
        """
        self.current_player = 0  # 重置当前玩家为 0
        for _ in range(self.num_playouts):
            state_copy = copy.deepcopy(state)  # 深拷贝当前状态
            env_copy = copy.deepcopy(env)  # 深拷贝当前环境
            self.perform_playout(state_copy, env_copy, self.play_depth)  # 执行模拟

        if not self.root.children:
            return None  # 如果没有可选动作，返回 None

        # 选择访问次数最多的子节点的动作
        best_action = max(self.root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_action  # 返回最佳动作

    def update_tree(self, last_action):
        """
        更新树结构，假设已经调用过 choose_action()。

        :param last_action: 上一个动作。
        """
        if last_action in self.root.children:
            self.root = self.root.children[last_action]  # 更新根节点为上一个动作的子节点
            self.root.parent = None  # 清除父节点
        else:
            self.root = Node()  # 如果没有找到，重置根节点

