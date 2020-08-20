#!/usr/bin/env python

import os, sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import random
from parse_osm import parser
from osm2graph import convert2graph, shortest_path, get_subgraph
import sumo_handler as sh
from practice import Network, DQN

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environmentvariable 'SUMO_HOME'")

import traci

routes = []
while True:
    source = random.choice(list(sh.osm_graph))
    target = random.choice(list(sh.osm_graph))

    try:
        route = sh.generateRoute(source, target)
    except:
        continue

    routes.append(route)

    if len(routes) == 1000:
        break

rl_env = sh.env(sh.osm_graph, routes, sh.tls)
# environment parameters
n_actions = rl_env.n_actions
state_shape = rl_env.state_shape
# hyper parameters
batch_size = 20
lr = 0.01
epsilon = 0.1
gamma = 0.9
target_replace_iter = 100
buffer_capacity = 400
n_episode = 1000

# build DQN
dqn = DQN(state_shape, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, buffer_capacity)
# machine learning
for i_episode in range(n_episode):
    traci.start(['/home/sheng/git/sumo/bin/sumo', '-c', '/home/sheng/git/Ambulance-Routing/net_files/run.sumo.cfg'])
    t = 0
    rewards = 0
    state = rl_env.reset()
    while True:
        # choose one action
        action = dqn.choose_action(state)
        next_state, reward, done = rl_env.step(action)
        # save experience
        dqn.buffer(state, action, reward, next_state)
        # accumulate reward
        rewards += reward
        if dqn.buffer_counter > buffer_capacity:
            dqn.run()
        state = next_state

        if done:
            print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
            traci.close()
            break

        t += 1
