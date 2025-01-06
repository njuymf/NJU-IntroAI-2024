import random, collections
import torch
import numpy as np
import copy

StepOutput = collections.namedtuple("step_output", ["action", "probs"])


class Agent(object):
    def __init__(self):
        pass

    def step(self, timestep):
        raise NotImplementedError



