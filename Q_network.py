import gym
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

# build Network
class Network(nn.module):
    def __init__(self, n_state, n_hidden, n_action):
        super(Network, self).__init__()
        self.conv = nn.Conv2d(n_state, n_hidden, kernel_size=5, stride=1) # convolution
        self.bn1 = nn.BatchNorm2d(n_hidden) # normalization
        self.out = nn.Linear(n_hidden, n_action) # linearization

    def forward(self, x):
        x = F.relu(self.bn1(self.conv(x))) # use relu activation function
        action_value = self.out(x.view(x.size(0), -1)) # change multiple dimensions to one dimension for linear regression
        return action_value

# build Double Q-Network
# need two Network : evaluation_network & target_network
class DQN(object):
    def __init__(self, n_state, n_hidden, n_action, batch_size, epsilon, gamma, target_replace_time, memory_capacity):
        self.evaluation_net, target_net = Network(n_state, n_hidden, n_action), Network(n_state, n_hidden, n_action)

        self.memory = np.zeros((memory_capacity, n_state * 2 + 2)) # experience in each memory = state + next_state + reward + action
        self.optimizer = torch.optim.Adam(self.evaluation_net.parameters()) # torch.optim.Adam(model.parameters(), lr=0.01)
        self.loss_function = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0 # let target_network know when to update

        self.n_stste = n_state
        self.n_hidden = n_hidden
        self.n_action = n_action
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_time = target_replace_time
        self.memory_capacity = memory_capacity

    # epsilon-greedy policy
    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)
        # random choose one action
        if np.random.uniform() < self.epsilon: # random choose one number from [0, 1)
            action = np.random.randint(0, self.n_action) # random choose one integer from [0, n_action)
        # according to current policy, choose the best action
        esle:
            action_value = self.evaluation_net(x) # get score(action_value) for each action
            action = torch.max(action_value, 1)[1].data.numpy()[0] # pick the action which score is the highest
        return action

    # sample from memory then learn
    def learn(self):
        # random choose batch_size experiences from memory_capacity
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # definition
        batch_state = torch.FloatTensor(batch_memory[:, :self.n_state])
        batch_action = torch.LongTensor(batch_memory[:, self.n_state:self.n_state+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.n_state+1:self.n_state+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.n_state:])
        # calculate the Q-value difference between evaluation_net and target_net
        q_evaluation = self.evaluation_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_function(q_evaluation, q_target)
        # Backpropagation
        self.optimizer.zero_grad() # reset to aovid batch gradient accumulation
        loss.backward()
        self.optimizer.step()
        # every target_replace_time update parameters of target_network by copy those of evaluation_network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_time == 0:
            self.target_net.load_state_dict(self.evaluation_net.state_dict())



