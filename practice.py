#!/usr/bin/env python


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Q(n1_state, n_action) = n1_reward
class TNetwork(nn.Module):
    def __init__(self, n1_state, n_action):
        super(target_net, self).__init__() # nn.Module.__init__()
        self.conv = nn.Conv2d(n1_state, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.out = nn.Linear(16, n_action)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv(x)))
        n1_reward = self.out(x.view(x.size(0), -1))

    return n1_reward # [a0, a1, a2, a3]

# Q(n0_state, n_action) = n0_reward
class ENetwork(nn.Module):
    def __init__(self, n0_state, n_action):
        super(evaluation_net, self).__init__()
        self.conv = nn.Conv2d(n0_state, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.out = nn.Linear(16, n_action)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv(x)))
        n0_reward = self.out(x.view(x.size(0), -1))

    return n0_reward # [a0, a1, a2, a3]

class DQN(object):
    def __init__(self, n_action, buffer_capacity, batch_size):
        self.targrt_net = TNetwork(n1_state, n_action)
        self.evaluation_net = ENetwork(n0_state, n_action)

        self.buffer_counter = 0

        self.n_action = n_action
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

    # choose one action which has the maximum reward
    def choose_action(self, state):
        rewards = []
        for a in n_action:
            reward = evaluation_net(state, a)
            rewards.append(reward)
        best_action = n_action[np.argmax(rewards)]
        return best_action

    def choose_action(self, state):
        reward = evaluation_net(state, n_action)
        best_action = n_action[np.argmax(reward)]

    def buffer(self, state, action, reward, next_state):
        memory = []
        transition = [state, action, reward, next_state]
        index = self.buffer_counter % self.buffer_capacity
        memory.insert(index, transition)
        self.buffer_counter += 1

    def run(self):
        # random choose batch_size experiences from buffer
        sample_index = np.random.choice(self.buffer_capacity, self.batch_size, replace=False).tolist()
        batch_memory, batch_state, batch_action, batch_reward, batch_next_state = [], [], [], [], []
        for i in sample_index:
            batch_memary.append(memory[i])
            # batch_state.append(memory[i][0])
            QEvaluation = self.evaluation_net(memory[i][0])
            # batch_next_state.append(memory[i][3])
            QNext = self.target_net(memory[i][3])
            QNextMax = QNext[np.argmax(QNext)]
            # batch_reward.append(memory[i][2])
            QTarget = QNextMax + memory[i][2]
            # batch_action.append(memory[i][1])
            # batch_reward.append(memory[i][2])




