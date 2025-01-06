import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("../")
sys.path.append("../environment")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from absl import logging, flags, app
from environment.GoEnv import Go

def get_max_idx(path):
    all_models = []
    for i in list(os.walk(path))[-1][-1]:
        all_models.append(i.split(".")[0])
    max_idx = max([eval(i) for i in all_models if i.isdigit()])

    return max_idx
# 这里演示如何在 PyTorch 中实现 PolicyGradient 和 DQN 的替代模块
# 您可以根据需要进行修改和扩展
class PolicyModule(nn.Module):
    """
    使用卷积层 + 全连接层的简单示例，替代原先的 TF PolicyGradient
    """
    def __init__(self, board_size, num_actions, hidden_layers_sizes=None, cnn_parameters=None):
        super().__init__()
        if hidden_layers_sizes is None:
            hidden_layers_sizes = [32, 64, 14]
        # cnn_parameters 格式： [output_channels, kernel_shapes, strides, paddings]
        if not cnn_parameters:
            cnn_parameters = [
                [2, 4, 8],
                [3, 3, 3],
                [1, 1, 1],
                ["SAME", "SAME", "VALID"]
            ]
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # 假设输入是单通道
        out_channels_list, kernel_shapes, strides, paddings = cnn_parameters
        # 构建卷积层
        for out_ch, k, s, pad in zip(out_channels_list, kernel_shapes, strides, paddings):
            # PyTorch 没有 "SAME"/"VALID" 关键字，需要根据需要自行计算 padding
            # 这里只是演示
            if pad == "SAME":
                padding_val = k // 2
            else:
                padding_val = 0

            self.conv_layers.append(
                nn.Conv2d(in_channels, out_ch, kernel_size=k, stride=s, padding=padding_val)
            )
            in_channels = out_ch
        conv_out_dim = out_channels_list[-1]  # 假设最后一次卷积的输出尺度可以被整成 1D
        # 根据 hidden_layers_sizes 构建全连接
        fc_layers = []
        flatten_dim = conv_out_dim * (board_size - 2) * (board_size - 2)
        input_dim = flatten_dim
        for h in hidden_layers_sizes:
            fc_layers.append(nn.Linear(input_dim, h))
            fc_layers.append(nn.ReLU())
            input_dim = h

        # 最后输出到 num_actions
        fc_layers.append(nn.Linear(input_dim, num_actions))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x shape: (batch_size, board_size, board_size)
        # 先扩一维，再通过 conv
        x = x.unsqueeze(1).float()
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
        # 展平并送入全连接
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def restore(self, path):
        self.load_state_dict(torch.load(path))

class DQNModule(nn.Module):
    """
    简单示例 DQN，替代原先的 TF DQN。
    """
    def __init__(self, input_size, num_actions, hidden_layers_sizes=[128,128]):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_layers_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())

    def save(self, path):
        torch.save(self.state_dict(), path)

    def restore(self, path):
        self.load_state_dict(torch.load(path))

class MCTSAgent:
    """
    演示用的MCTSAgent，实现上可以随意修改，但对外接口保持不变。
    """
    def __init__(self, policy_module, rollout_module, playout_depth=10, n_playout=100):
        """
        policy_module: 主策略网络
        rollout_module: 回合模拟网络
        """
        self.policy_module = policy_module
        self.rollout_module = rollout_module
        self.playout_depth = playout_depth
        self.n_playout = n_playout
        # 用来记录损失等信息
        self.loss = 0

    def step(self, time_step, env):
        """
        对外提供的动作决策接口，输入游戏状态，输出要执行的动作
        """
        # 以下仅演示。实际中应结合蒙特卡洛树搜索进行动作选择
        # 如果没有 policy_module，就随机出一个动作
        if self.policy_module is None:
            action = np.random.choice(env.action_size)
            return [action]
        
        # 有策略网络时，就简单调用网络做个输出
        with torch.no_grad():
            obs = torch.tensor(time_step.observations['info_state'][0]).unsqueeze(0)
            logits = self.policy_module(obs)
            action = torch.argmax(logits, dim=-1).item()
        return [action]

    def save(self, checkpoint_root, checkpoint_name='agent'):
        """
        保存模型
        """
        if self.policy_module is not None:
            model_path = os.path.join(checkpoint_root, checkpoint_name + "_policy.pt")
            self.policy_module.save(model_path)
        if self.rollout_module is not None:
            rollout_path = os.path.join(checkpoint_root, checkpoint_name + "_rollout.pt")
            self.rollout_module.save(rollout_path)

    def restore(self, path):
        """
        加载模型
        """
        if self.policy_module is not None:
            policy_path = path + "_policy.pt"
            self.policy_module.restore(policy_path)
        if self.rollout_module is not None:
            rollout_path = path + "_rollout.pt"
            self.rollout_module.restore(rollout_path)

###
# 以下部分主要是原脚本中功能的 PyTorch 版本改写
###

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", 200000,
                     "Number of training episodes for each base policy.")
flags.DEFINE_integer("num_eval", 1000,
                     "Number of evaluation episodes")
flags.DEFINE_integer("eval_every", 2000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("learn_every", 128,
                     "Episode frequency at which the agents learn.")
flags.DEFINE_integer("save_every", 5000,
                     "Episode frequency at which the agents save the policies.")
flags.DEFINE_list("output_channels", [
    2, 4, 8, 16, 32
], "")
flags.DEFINE_list("hidden_layers_sizes", [
    32, 64, 14
], "Number of hidden units in the net.")
flags.DEFINE_integer("replay_buffer_capacity", int(5e4),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_bool("use_dqn", False, "use dqn or not. If set to false, use a2c")
flags.DEFINE_float("lr", 2e-4, "lr")
flags.DEFINE_integer("pd", 10, "playout_depth")
flags.DEFINE_integer("np", 100, "n_playout")


def use_dqn():
    return FLAGS.use_dqn

def fmt_hyperparameters():
    fmt = ""
    for i in FLAGS.output_channels:
        fmt += '_{}'.format(i)
    fmt += '**'
    for i in FLAGS.hidden_layers_sizes:
        fmt += '_{}'.format(i)
    return fmt

def init_env():
    begin = time.time()
    env = Go(flatten_board_state=False)
    info_state_size = env.state_size
    print(info_state_size)
    num_actions = env.action_size
    return env, info_state_size, num_actions, begin

def init_hyper_paras():
    num_cnn_layer = len(FLAGS.output_channels)
    kernel_shapes = [3 for _ in range(num_cnn_layer)]
    strides = [1 for _ in range(num_cnn_layer)]
    paddings = ["SAME" for _ in range(num_cnn_layer - 1)]
    paddings.append("VALID")
    cnn_parameters = [FLAGS.output_channels, kernel_shapes, strides, paddings]
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    # 这里示意使用 DQN / A2C 对应的 kwargs
    dqn_kwargs = {
        "hidden_layers_sizes": [128, 128],
        "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
        "epsilon_decay_duration": int(0.6 * FLAGS.num_train_episodes),
        "epsilon_start": 0.8,
        "epsilon_end": 0.001,
        "learning_rate": FLAGS.lr,
        "learn_every": FLAGS.learn_every,
        "batch_size": 256,
    }
    a2c_kwargs = {
        "hidden_layers_sizes": hidden_layers_sizes,
        "cnn_parameters": cnn_parameters,
        "pi_learning_rate": 3e-4,
        "critic_learning_rate": 1e-3,
        "batch_size": 128,
        "entropy_cost": 0.5,
    }
    return dqn_kwargs, a2c_kwargs

def init_agents(sess,
                info_state_size,
                num_actions,
                dqn_kwargs,
                a2c_kwargs):
    board_size = int(info_state_size**0.5)
    if use_dqn():
        # 如果选择 DQN
        policy_module = DQNModule(info_state_size, num_actions, hidden_layers_sizes=dqn_kwargs["hidden_layers_sizes"])
        rollout_module = DQNModule(info_state_size, num_actions, hidden_layers_sizes=dqn_kwargs["hidden_layers_sizes"])
    else:
        # 如果选择 A2C / policy gradient
        policy_module = PolicyModule(board_size, num_actions,
                                     hidden_layers_sizes=a2c_kwargs["hidden_layers_sizes"],
                                     cnn_parameters=a2c_kwargs["cnn_parameters"])
        rollout_module = PolicyModule(board_size, num_actions,
                                      hidden_layers_sizes=a2c_kwargs["hidden_layers_sizes"],
                                      cnn_parameters=a2c_kwargs["cnn_parameters"])
    for param in policy_module.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    for param in rollout_module.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    # 将 policy_module 的权重复制到 rollout_module
    rollout_module.load_state_dict(policy_module.state_dict())
    agents = [
        MCTSAgent(policy_module, rollout_module,
                  playout_depth=FLAGS.pd, n_playout=FLAGS.np),
        MCTSAgent(None, None)  # 第二个 Agent 只随机
    ]
    logging.info("MCTS INIT OK!!")
    return agents


def prt_logs(ep, agents, ret, begin):
    losses = agents[0].loss
    logging.info("Episodes: {}: Losses: {}, Rewards: {}".format(ep + 1, losses, np.mean(ret)))

    alg_tag = "dqn_cnn_vs_rand" if use_dqn() else "a2c_cnn_vs_rnd"
    with open('../logs/log_{}_{}'.format(os.environ.get('BOARD_SIZE'), alg_tag + fmt_hyperparameters()), 'a+') as log_file:
        log_file.writelines("{}, {}\n".format(ep + 1, np.mean(ret)))


def save_model(ep, agents):
    alg_tag = "../saved_model/CNN_DQN" if use_dqn() else "../saved_model/CNN_A2C"
    save_path = alg_tag + fmt_hyperparameters()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    agents[0].save(checkpoint_root=save_path, checkpoint_name='{}'.format(ep + 1))
    print("Model Saved!")


def restore_model(agents, path=None):
    alg_tag = "../saved_model/CNN_DQN" if use_dqn() else "../saved_model/CNN_A2C"
    try:
        if path:
            agents[0].restore(path)
            idex = path.split("/")[-1]
        else:
            idex = get_max_idx(alg_tag + fmt_hyperparameters())
            path = os.path.join(alg_tag + fmt_hyperparameters(), str(idex))
            agents[0].restore(path)
        logging.info("Agent model restored at {}".format(path))
    except:
        print(sys.exc_info())
        logging.info("Train From Scratch!!")
        idex = 0
    return int(idex)


def evaluate(agents, env):
    ret = []
    for ep in range(FLAGS.num_eval):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step, env)
            time_step = env.step(agent_output)
        logging.info(time_step.rewards)
        ret.append(time_step.rewards[0])
    return ret


def stat(ret, begin):
    print(np.mean(ret))
    print('Time elapsed:', time.time() - begin)


def main(unused_argv):
    env, info_state_size, num_actions, begin = init_env()
    dqn_kwargs, a2c_kwargs = init_hyper_paras()
    sess = None

    agents = init_agents(sess, info_state_size, num_actions, dqn_kwargs, a2c_kwargs)
    ret = evaluate(agents, env)
    stat(ret, begin)


if __name__ == '__main__':
    app.run(main)