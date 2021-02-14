from model.dqn_agent import DQNAgent
import gym
import torch
import numpy as np

env_id = "MountainCar-v0"
env = gym.make(env_id)

seed = 777


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
seed_torch(seed)
env.seed(seed)


# parameters
num_frames = 100000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 1 / 2000

agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
agent.train(num_frames)
agent.test()
