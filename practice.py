#!/usr/bin/env python


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Q(n1_state, n_action) = n1_reward
class TNetwork(nn.Module):
    def __init__(self, n1_state, n1_action):
        super(target_net, self).__init__() # nn.Module.__init__()
        self.conv = nn.Conv2d(n1_state, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.out = nn.Linear(16, n_action)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv(x)))
        n1_reward = self.out(x.view(x.size(0), -1)) # [a0, a1, a2, a3]
        # softmax
        exp_n1 = np.exp(n1_reward)
        softmax_n1 = exp_n1 / np.sum(exp_n1)

    return softmax_n1

# Q(n0_state, n_action) = n0_reward
class ENetwork(nn.Module):
    def __init__(self, n0_state, n_action):
        super(evaluation_net, self).__init__()
        self.conv = nn.Conv2d(n0_state, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.out = nn.Linear(16, n_action)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv(x)))
        n0_reward = self.out(x.view(x.size(0), -1)) # [a0, a1, a2, a3]
        # softmax
        exp_n0 = np.exp(n0_reward)
        softmax_n0 = exp_n0 / np.sum(exp_n0)

    return softmax_n0

class DQN(object):
    def __init__(self, n_action, batch_size, buffer_capacity):
        self.targrt_net = TNetwork(n1_state, n_action)
        self.evaluation_net = ENetwork(n0_state, n_action)

        self.buffer_counter = 0
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.evaluation_net.parameters())

        self.n_action = n_action
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

    # choose one action which has the maximum reward
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
        state, action, reward, next_state = # function call
        # Evaluation_Qvalue = Q(st, at)
        Evaluation_Qvalue = self.evaluation_net(state, action)
        # V(st+1, at+1) = maxQ(st+1, a)
        V_next_state = self.target_net(next_state, n_action)
        V_max_next_state = V_next_state[np.argmax(V_next_state)]
        # Target_Qvalue = V(st+1, at+1) + reward
        Target_Qvalue = V_max_next_state + reward
        # calculate loss between Evaluation_Qvalue and Target_Qvalue
        loss = self.loss_function(Evaluation_Qvalue, Target_Qvalue)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




