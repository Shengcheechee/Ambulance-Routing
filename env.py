#!/usr/local/bin/python3
import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environmentvariable 'SUMO_HOME'")
import traci
import sumolib

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
        edges[edge.get('id').split("#")[0]] = []

    found = False
    tmp = ""
    edges_info = list(sumo_graph.edges.data('id'))
    edges_info = sorted(edges_info, key = lambda e: int(e[2].split("#")[1]) if len(e[2].split("#")) > 1 else 0)
    edges_info = sorted(edges_info, key = lambda e: e[2].split("#")[0])

    for node in list(osm_graph):
        for edge in edges_info:
            if edge[2].split("#")[0] != tmp and tmp:
                found = False
                path = []

            if edge[0] == node:
                tmp = edge[2].split("#")[0]
                found = True
                path = [node]

            if found:
                path.append(edge[2])

            if found and edge[1] in list(osm_graph):
                tmp = ""
                found = False
                path.append(edge[1])
                path = tuple(path)
                edges[edge[2].split("#")[0]].append(path)

    return sumo_graph, edges

def get_generateRoute(sumo_graph):
    def generateRoute(src, tgt):

        return shortest_path(sumo_graph, src, tgt)

    return generateRoute

def	addCar(route):
	traci.route.add(routeID="newRoute", edges=route)
	traci.vehicle.add(vehID = "ambulance", routeID = route, typeID = "Amb")
	traci.vehicle.setVehicleClass(vehID = "ambulance", clazz = "emergency")

def get_find_neighbor_nodes(edges):
    def find_neighbor_nodes(path):
        for edge in edges.keys():
            if path in edges[edge]:
                return edges[edge][0], edges[edge][-1]
    return find_neighbor_nodes

class env(object):
    def __init__(self, osm_graph, routes):
        self.graph = osm_graph
        self.routes = routes
        self.n_actions = 4
        self.count = 0

    def reset(self):
        addCar(self.routes[count])
        self.start_time = traci.simulation.getTime
        n1, n2 = find_neighbor_nodes(self.routes[count][0])
        state = gnn(get_subgraph(self.graph, n1. n2))
        return state

    def step(self, action):
        def perform_action(action):
            if action == 1:
                for node in nx_graph:
                    if nx.get_node_attributes(nx_graph, 'trafsig')[node]:
                        traci.trafficlight.setRedYellowGreenState(node, 'GGGG')

            elif action == 2:
                for node in nx_graph:
                    if nx.get_node_attributes(nx_graph, 'trafsig')[node]:
                        traci.trafficlight.setRedYellowGreenState(node, 'RRRR')

            elif action == 3:
                for node in nx_graph:
                    if nx.get_node_attributes(nx_graph, 'trafsig')[node]:
                        traci.trafficlight.setRedYellowGreenState(node, 'GGGGGGYyRR')

            elif action == 4:
                for node in nx_graph:
                    if nx.get_node_attributes(nx_graph, 'trafsig')[node]:
                        traci.trafficlight.setRedYellowGreenState(node, 'RRRRRRGGYy')

            else:
                pass

        def get_reward():
            travel_time = traci.simulation.getTime() - self.start_time
            lane_vehicles_number = traci.lane.getLastStepVehicleNumber(traci.vehicle.getLaneID("ambulance"))
            reward = 5 - lan_vehicle_number
            return reward

        def get_next_state():
            n1, n2 = find_neighbor_nodes(traci.vehicle.getRoadID())
            state = gnn(get_subgraph(self.graph, n1. n2))
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
    nodes, ways = parser('ncku.osm')
    osm_graph = convert2graph(nodes, ways)
    source = "5520434289"
    target = "355961064"
    sumo_graph, edges = get_sumo_info('ncku.net.xml', osm_graph)
    generateRoute = get_generateRoute(sumo_graph)
    route = generateRoute(source, target)
    print("The route is :", route)
