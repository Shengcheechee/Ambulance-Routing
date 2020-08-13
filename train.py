#!/usr/bin/env python


import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import practice.py


env = # environment
# environment parameters
n_action = [0, 1, 2 ,3]
n_state = np.zeros((batch_size, 32, 32))
# hyper parameters
batch_size = 20
buffer_capacity = 2000
n_episode = 4000

# build DQN
dqn = DQN(n_action, batch_size, buffer_capacity)
# machine learning
for i_episode in range(n_episode):
    rewards = 0
    state = env.reset()
    while True:
        # choose one action
        action = dqn.choose_action(state)
        next_state, reward, done = env.step(action)
        # save experience
        dqn.buffer(state, action, reward, next_state)
        # accumulate reward
        rewards += reward
        if dqn.buffer_counter > buffer_capacity:
            dqn.run()
        state = next_state

env.close()





