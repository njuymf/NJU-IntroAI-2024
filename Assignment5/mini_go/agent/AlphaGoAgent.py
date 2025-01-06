
import random  # 导入随机模块用于随机选择
import numpy as np  # 导入NumPy进行数值运算
import copy  # 导入copy模块以实现深拷贝
import collections  # 导入collections以使用命名元组

from agent.agent import Agent  # 从agent模块导入基础Agent类

# 定义一个命名元组来存储每一步的行动和概率
StepResult = collections.namedtuple("step_result", ["action", "probability"])
StepOutput = collections.namedtuple("step_output", ["action", "probs"])


class MiniAlphaGoAgent(Agent):
    def __init__(self, _id, mini_alphago):
        super().__init__()
        self.player_id = _id
        self.mini_alphago = mini_alphago


    def step(self, state,env,timestep):
        chosen_action,p= self.mini_alphago.step(state,env,timestep)
        return StepOutput(action=chosen_action, probs=p)