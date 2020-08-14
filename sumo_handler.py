#!/usr/local/bin/python3
import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environmentvariable 'SUMO_HOME'")

import traci
import xml.etree.ElementTree as ET
import cv2
import networkx as nx
import numpy as np
import random
from parse_osm import parser, render
from osm2graph import convert2graph, shortest_path, get_subgraph

def relu(x):
    return np.maximum(x, 0)

def layer(A, D, F):
    return relu(D ** -1 * A * F)

def weighted_function(node, nx_graph, A):
    weight = 0
    trafsig = nx.get_node_attributes(nx_graph, 'trafsig')

    if trafsig[node]:
        weight = weight + 1000
    for nd in A[node].keys():
        weight = weight + A[node][nd]['weight']

    return weight

def gnn(nx_graph):
    A = nx.to_dict_of_dicts(nx_graph)
    A_ = sorted(A, key = lambda nd: weighted_function(nd, nx_graph, A), reverse = True)
    sorted_A = nx.to_numpy_matrix(nx_graph, nodelist = A_)
    I = np.eye(nx_graph.number_of_nodes())
    A_self = sorted_A + I

    D_array = np.array(np.sum(A_self, axis = 1)).reshape(-1)
    D = np.matrix(np.diag(D_array))

    F = []
    for node in A_:
        F.append([int(nx.get_node_attributes(nx_graph, 'trafsig')[node])])

    H_1 = layer(A_self, D, F)
    H_2 = layer(A_self, D, H_1)

    return H_2

def replay_buffer(self, state, action, reward, next_state):

    transition = np.hstack((state, [action, reward], next_state))

    index = self.memory_cuounter % self.memory_capacity
    self.memory[index, :] = transition
    self.memory_counter += 1

def get_sumo_info(netfile, osm_graph):
    root = ET.parse(netfile).getroot()
    sumo_graph = nx.DiGraph()

    edges = {}
    for edge in root.findall('edge'):
        sumo_graph.add_edge(edge.get('from'), edge.get('to'), id = edge.get('id'))
        if edge.get('from'):
            edges[str(abs(int(edge.get('id').split("#")[0])))] = []

    found = False
    tmp = ""
    edges_info = list(sumo_graph.edges.data('id'))
    edges_info = sorted(edges_info, key = lambda e: int(e[2].split("#")[1]) if len(e[2].split("#")) > 1 else 0)
    edges_info = sorted(edges_info, key = lambda e: e[2].split("#")[0])

    for node in list(osm_graph):
        for edge in edges_info:
            if edge[2].split("#")[0][0] == '-':
                continue
            else:
                edge_id = edge[2].split("#")[0]

            if edge_id != tmp and tmp:
                found = False
                path = []

            if edge[0] == node:
                tmp = edge_id
                found = True
                path = [node]

            if found:
                path.append(edge[2])

            if found and edge[1] in list(osm_graph):
                tmp = ""
                found = False
                path.append(edge[1])
                path = tuple(path)
                edges[edge_id].append(path)

    return sumo_graph, edges

def get_generateRoute(sumo_graph):
    def generateRoute(src, tgt):

        return shortest_path(sumo_graph, src, tgt)

    return generateRoute

def	addCar(route):
	traci.route.add(routeID="newRoute", edges=route)
	traci.vehicle.add(vehID = "ambulance", routeID = "newRoute")
	traci.vehicle.setVehicleClass(vehID = "ambulance", clazz = "ignoring")

def get_find_neighbor_nodes(edges):
    def find_neighbor_nodes(path):
        edge_id = str(abs(int(path.split("#")[0])))
        path = edge_id + "#" + path.split("#")[1] if len(path.split("#")[1]) > 1 else edge_id

        for i in range(len(edges[edge_id])):
            if path in edges[edge_id][i]:
                return edges[edge_id][i][0], edges[edge_id][i][-1]
            else:
                pass

    return find_neighbor_nodes

class env(object):
    def __init__(self, osm_graph, routes):
        self.graph = osm_graph
        self.routes = routes
        self.n_actions = 4
        self.count = 0

    def reset(self):
        addCar(self.routes[self.count])
        print(traci.vehicle.getRoadID("ambulance"))
        print(traci.vehicle.getIDList())
        traci.simulationStep()
        print(traci.vehicle.getRoadID("ambulance"))
        print(traci.vehicle.getIDList())
        traci.simulationStep()
        print(traci.vehicle.getRoadID("ambulance"))
        self.start_time = traci.simulation.getTime()
        n1, n2 = find_neighbor_nodes(self.routes[self.count][0])
        self.subgraph = get_subgraph(self.graph, n1, n2)
        state = gnn(self.subgraph)
        return state

    def step(self, action):
        def perform_action(action):
            if action == 1:
                # for node in self.subgraph:
                #     if nx.get_node_attributes(self.subgraph, 'trafsig')[node]:
                #         print(traci.trafficlight.getCompleteRedYellowGreenDefinition(node))
                #         traci.trafficlight.setPhase(node, )
                pass
            elif action == 2:
                # for node in self.subgraph:
                #     if nx.get_node_attributes(self.subgraph, 'trafsig')[node]:
                #         traci.trafficlight.setPhase(node, )
                pass
            elif action == 3:
                # for node in self.subgraph:
                #     if nx.get_node_attributes(self.subgraph, 'trafsig')[node]:
                #         traci.trafficlight.setPhase(node, )
                pass
            elif action == 4:
                # for node in self.subgraph:
                #     if nx.get_node_attributes(self.subgraph, 'trafsig')[node]:
                #         traci.trafficlight.setPhase(node, )
                pass
            else:
                pass

        def get_reward():
            travel_time = traci.simulation.getTime() - self.start_time
            lane_vehicles_number = traci.lane.getLastStepVehicleNumber(traci.vehicle.getLaneID("ambulance"))
            reward = 5 - lane_vehicles_number
            return reward

        def get_next_state():
            print(traci.vehicle.getRoute("ambulance"))
            print(traci.vehicle.getLaneID("ambulance"))
            n1, n2 = find_neighbor_nodes(traci.vehicle.getRoadID("ambulance"))
            state = gnn(get_subgraph(self.graph, n1, n2))
            return state

        def check_done():
            self.count += 1
            return "ambulance" in traci.simulation.getArrivedIDList()

        perform_action(action)
        done = check_done()
        reward = get_reward()

        traci.simulation.saveState('now')
        traci.simulationStep()
        next_state = get_next_state()
        traci.simulation.loadState('now')

        return next_state, reward, done


if __name__ == '__main__':
    nodes, ways = parser('/home/sheng/git/Ambulance-Routing/net_files/ncku.osm')
    osm_graph = convert2graph(nodes, ways)
    source = "5520434289"
    target = "355961064"
    sumo_graph, edges = get_sumo_info('/home/sheng/git/Ambulance-Routing/net_files/ncku.net.xml', osm_graph)
    find_neighbor_nodes = get_find_neighbor_nodes(edges)
    generateRoute = get_generateRoute(sumo_graph)
    route = [f'-111343192#{i}' for i in range(11, 0, -1)]
    # route = generateRoute(source, target)
    print("The route is :", route)
    routes = [route]
    traci.start(['/home/sheng/git/sumo/bin/sumo', '-c', '/home/sheng/git/Ambulance-Routing/net_files/run.sumo.cfg'])
    rl_env = env(osm_graph, routes)
    s = rl_env.reset()
    n_s, r, d = rl_env.step(1)
    print("s :", s)
    print("n_s :", n_s)
    print("r :", r)
    print("d :", d)
    traci.close()
