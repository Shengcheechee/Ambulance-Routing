import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import Q_network.py

env = FakeEnv()

# environment parameters
n_action = env.action_space.n # four actions : add 2s/4s, subtract 2s/4s
n_state = env.observation_space.shape[0]
# hyper parameters
n_hidden = 50
batch_size = 32
epsilon = 0.1
gamma = 0.9 # reward discount factor
target_replace_time = 100
memory_capacity = 2000
n_episode = 4000

# build DQN
dqn = DQN(n_state, n_hidden, n_action, batch_size, epsilon, gamma, target_replace_time, memory_capacity)
# machine learning
for i_episode in range(n_episode):
    t =0
    rewards = 0
    state = env.reset()
    while True:
        # choose one action
        # choose action randomly at first time
        action = dqn.choose_action(state)
        next_state, reward, done, info = env.step(action)
        # save experience
        dqn.store_transition(state, action, reward, next_state)
        # accumulate reward
        rewards += reward

        if dqn.memory_counter > memory_capacity:
            dqn.learn()

        state = next_state
        t += 1
env.close()





