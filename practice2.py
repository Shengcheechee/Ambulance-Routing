#!/usr/bin/env python


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q(n1_state, n_action) = n1_reward
class Network(nn.Module):
    def __init__(self, w, h, n_actions):
        super(Network, self).__init__() # nn.Module.__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=5, stride = 1)
        self.bn1 = nn.BatchNorm2d(16)

        def conv2d_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(w)
        convh = conv2d_size_out(h)
        linear_input_size = convw * convh * 16
        self.out = nn.Linear(linear_input_size, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv(x)))
        n1_reward = self.out(x.view(x.size(0), -1)) # [a0, a1, a2, a3]
        # softmax
        # exp_n1 = np.exp(n1_reward)
        # softmax_n1 = exp_n1 / np.sum(exp_n1)

        return n1_reward

# R(n0_state, n_action) = n0_reward
# class ENetwork(nn.Module):
#     def __init__(self, n0_state, n_action):
#         super(evaluation_net, self).__init__()
#         self.conv = nn.Conv2d(n0_state, 16, kernel_size=5, stride=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.out = nn.Linear(16, n_action)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv(x)))
#         n0_reward = self.out(x.view(x.size(0), -1)) # [a0, a1, a2, a3]
#         # softmax
#         exp_n0 = np.exp(n0_reward)
#         softmax_n0 = exp_n0 / np.sum(exp_n0)
#
#     return softmax_n0

class DQN(object):
    def __init__(self, state_shape, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, buffer_capacity):
        target_net = Network(state_shape, state_shape, n_actions)#.to(device)
        evaluation_net = Network(state_shape, state_shape, n_actions)#.to(device)
        self.target_net = target_net.float()
        self.evaluation_net = evaluation_net.float()

        self.buffer_counter = 0
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.evaluation_net.parameters(), lr = lr)
        self.learn_step_counter = 0
        self.memory = []

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.buffer_capacity = buffer_capacity

    # choose one action which has the maximum reward
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            best_action = np.random.randint(0, self.n_actions)
        else:
            reward = self.evaluation_net(state)
            best_action = torch.max(reward, 1)[1].data.numpy()[0]
            # best_action = self.n_actions[np.argmax(reward)]

        return best_action

    def buffer(self, state, action, reward, next_state):
        transition = [state, action, reward, next_state]
        index = self.buffer_counter % self.buffer_capacity
        if len(self.memory) < self.buffer_capacity:
            self.memory.append(None)
        self.memory[index] = transition
        self.buffer_counter += 1

    def run(self):
        # random choose batch_size experiences from buffer
        sample_index = np.random.choice(self.buffer_capacity)
        sample = self.memory[sample_index]
        state, action, reward, next_state = samle[0], sample[1], sample[2], sample[3]
        # Evaluation_Qvalue = Q(st, at)
        Evaluation_Qvalue = self.evaluation_net(state)
        # V(st+1, at+1) = maxQ(st+1, a)
        V_next_state = self.target_net(next_state).detach()
        V_max_next_state = V_next_state[np.argmax(V_next_state)]
        # Target_Qvalue = V(st+1, at+1) + reward
        Target_Qvalue = self.gamma * V_max_next_state + reward
        # calculate loss between Evaluation_Qvalue and Target_Qvalue
        loss = self.loss_function(Evaluation_Qvalue, Target_Qvalue)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.evaluation_net.state_dict())


