#!/usr/local/bin/python3
import os, sys
import xml.etree.ElementTree as ET
import cv2
import networkx as nx
import numpy as np
import random
from parse_osm import parser, render
from osm2graph import convert2graph, shortest_path, get_subgraph

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environmentvariable 'SUMO_HOME'")

import traci

# def relu(x):
#     return np.maximum(x, 0)
#
# def layer(A, D, F):
#     return relu(D ** -1 * A * F)
#
def weighted_function(node, nx_graph, A):
    weight = 0
    trafsig = nx.get_node_attributes(nx_graph, 'trafsig')

    if trafsig[node]:
        weight = weight + 1000
    for nd in A[node].keys():
        weight = weight + A[node][nd]['weight']

    return weight
#
# def gnn(nx_graph):
#     A = nx.to_dict_of_dicts(nx_graph)
#     A_ = sorted(A, key = lambda nd: weighted_function(nd, nx_graph, A), reverse = True)
#     sorted_A = nx.to_numpy_matrix(nx_graph, nodelist = A_)
#     I = np.eye(nx_graph.number_of_nodes())
#     A_self = sorted_A + I
#
#     D_array = np.array(np.sum(A_self, axis = 1)).reshape(-1)
#     D = np.matrix(np.diag(D_array))
#
#     F = []
#     for node in A_:
#         F.append([int(nx.get_node_attributes(nx_graph, 'trafsig')[node])])
#
#     H_1 = layer(A_self, D, F)
#     H_2 = layer(A_self, D, H_1)
#
#     return H_2

def get_adjmtx(nx_graph):
    A = nx.to_dict_of_dicts(nx_graph)
    A_ = sorted(A, key = lambda nd: weighted_function(nd, nx_graph, A), reverse = True)
    sorted_A = nx.to_numpy_matrix(nx_graph, nodelist = A_)
    I = np.eye(nx_graph.number_of_nodes())
    A_self = sorted_A + I

    if len(A_self) >= 32:
        adjmtx = A_self[: 32, : 32]
    else:
        add = 32 - len(A_self)
        adjmtx = A_self
        for i in range(add):
            adjmtx = np.r_[adjmtx, [np.zeros(len(adjmtx), dtype = int)]]
            adjmtx = np.c_[adjmtx, np.array(np.zeros(len(adjmtx), dtype = int)).T]

    return adjmtx

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
        if edge.get('type') and edge.get('type').split(".")[1] not in ["path", "footway", "steps", "pedestrian"]:
            sumo_graph.add_edge(edge.get('from'), edge.get('to'), id = edge.get('id'))
        if edge.get('from'):
            edges[str(abs(int(edge.get('id').split("#")[0])))] = []

    connections = {}
    for connection in root.findall('connection'):
        if connection.get('via'):
            connections[connection.get('via').rsplit('_', 1)[0]] = (connection.get('from'), connection.get('to'))

    tls = {}
    for tl in root.findall('tlLogic'):
        tls[tl.get('id')] = len(tl.find('phase').get('state'))

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

    return sumo_graph, edges, connections, tls

def get_generateRoute(sumo_graph):
    def generateRoute(src, tgt):

        return shortest_path(sumo_graph, src, tgt)

    return generateRoute

def	addCar(route):
	traci.route.add(routeID="newRoute", edges=route)
	traci.vehicle.add(vehID = "ambulance", routeID = "newRoute")
	traci.vehicle.setVehicleClass(vehID = "ambulance", clazz = "ignoring")

def get_find_neighbor_nodes(edges, connections):
    def find_neighbor_nodes(path):
        if '_' in path:
            path = connections[path][1]

        if path[0] == '-':
            path = path[1:]

        edge_id = path.split("#")[0]

        for i in range(len(edges[edge_id])):
            if path in edges[edge_id][i]:
                return edges[edge_id][i][0], edges[edge_id][i][-1]
            else:
                pass

    return find_neighbor_nodes

class env(object):
    def __init__(self, osm_graph, routes, tls):
        self.graph = osm_graph
        self.routes = routes
        self.n_actions = 4
        self.tls = tls
        self.count = 0

    def reset(self):
        addCar(self.routes[self.count])
        traci.simulationStep()
        self.start_time = traci.simulation.getTime()
        n1, n2 = find_neighbor_nodes(self.routes[self.count][0])
        self.subgraph = get_subgraph(self.graph, n1, n2)
        state = get_adjmtx(self.subgraph)
        return state

    def step(self, action):
        def perform_action(action):
            n1, n2 = find_neighbor_nodes(traci.vehicle.getRoadID("ambulance"))
            self.subgraph = get_subgraph(self.graph, n1, n2)

            if action == 1:
                for node in self.subgraph:
                    if node in self.tls:
                        traci.trafficlight.setRedYellowGreenState(node, 'G' * self.tls[node])

            elif action == 2:
                for node in self.subgraph:
                    if node in self.tls:
                        traci.trafficlight.setRedYellowGreenState(node, 'r' * self.tls[node])

            elif action == 3:
                for node in self.subgraph:
                    if node in self.tls:
                        traci.trafficlight.setRedYellowGreenState(node, 'G' * int(self.tls[node] * 0.6 + 0.5) + 'y' * int(self.tls[node] * 0.1 + 0.5) + 'r' * int(self.tls[node] * 0.3 + 0.5))

            elif action == 4:
                for node in self.subgraph:
                    if node in self.tls:
                        traci.trafficlight.setRedYellowGreenState(node, 'r' * int(self.tls[node] * 0.6 + 0.5) + 'y' * int(self.tls[node] * 0.1 + 0.5) + 'G' * int(self.tls[node] * 0.3 + 0.5))

            else:
                pass

        def get_reward():
            travel_time = traci.simulation.getTime() - self.start_time
            lane_vehicles_number = traci.lane.getLastStepVehicleNumber(traci.vehicle.getLaneID("ambulance"))
            reward = 5 - lane_vehicles_number
            return reward

        def get_next_state():
            n1, n2 = find_neighbor_nodes(traci.vehicle.getRoadID("ambulance"))
            state = get_adjmtx(get_subgraph(self.graph, n1, n2))
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
    source = random.choice(list(osm_graph))
    target = random.choice(list(osm_graph))
    sumo_graph, edges, connections, tls = get_sumo_info('/home/sheng/git/Ambulance-Routing/net_files/ncku.net.xml', osm_graph)
    find_neighbor_nodes = get_find_neighbor_nodes(edges, connections)
    generateRoute = get_generateRoute(sumo_graph)
    route = generateRoute(source, target)
    print("The route is :", route)
    routes = [route]
    traci.start(['/home/sheng/git/sumo/bin/sumo', '-c', '/home/sheng/git/Ambulance-Routing/net_files/run.sumo.cfg'])
    rl_env = env(osm_graph, routes, tls)
    s = rl_env.reset()
    n_s, r, d = rl_env.step(1)
    print("s :", s)
    print("n_s :", n_s)
    print("r :", r)
    print("d :", d)
    traci.close()
