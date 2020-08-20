#!/usr/local/bin/python3
import os, sys
import math
import random
from parse_osm import parser
from osm2graph import convert2graph, shortest_path, get_subgraph
import sumo_handler as sh

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

    if len(routes) == 30:
        break

non_rl_env = sh.env(sh.osm_graph, routes, sh.tls)
total_time = 0

for i in range(30):
    traci.start(['/home/sheng/git/sumo/bin/sumo', '-c', '/home/sheng/git/Ambulance-Routing/net_files/run.sumo.cfg'])
    non_rl_env.start()
    while True:
        done, travel_time = non_rl_env.random_step()

        if done:
            total_time += travel_time
            print(f"Episode finished after {travel_time} seconds.")
            traci.close()
            break

average_time = total_time / 30
print(f"Average travel time is : {average_time}")
