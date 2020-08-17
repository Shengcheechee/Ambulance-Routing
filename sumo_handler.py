#!/usr/local/bin/python3
import os, sys
import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import random
from parse_osm import parser
from osm2graph import convert2graph, shortest_path, get_subgraph

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environmentvariable 'SUMO_HOME'")

import traci

def get_adjmtx(nx_graph):
    A = nx.to_numpy_matrix(nx_graph)
    I = np.eye(nx_graph.number_of_nodes())
    A_self = A + I

    # cut the adjancency matrix to the shape of 32x32
    # and return 32 x 32 matrix as state

    if len(A_self) >= 32:
        adjmtx = A_self[: 32, : 32]
    else:
        add = 32 - len(A_self)
        adjmtx = A_self
        for i in range(add):
            adjmtx = np.r_[adjmtx, [np.zeros(len(adjmtx), dtype = int)]]
            adjmtx = np.c_[adjmtx, np.array(np.zeros(len(adjmtx), dtype = int)).T]

    return adjmtx

def get_sumo_info(netfile, osm_graph):
    root = ET.parse(netfile).getroot()
    sumo_graph = nx.DiGraph()

    # mapping the sumo_graph edges to the osm_graph edges
    # edges = {
    #     "edge_id_in_osm": [(junction_A, edge_id_in_sumo_1, edge_id_in_sumo_2, junction_B),
    #                        (junction_C, edge_id_in_sumo_3, junction_D), ...]
    #     ...
    # }

    edges = {}
    for edge in root.findall('edge'):

        # create sumo_graph
        if edge.get('type') and edge.get('type').split(".")[1] not in ["path", "footway", "steps", "pedestrian"]:
            sumo_graph.add_edge(edge.get('from'), edge.get('to'), id = edge.get('id'))

        # create a dict only has keys of osm_edge_ids without considering direction
        if edge.get('from'):
            edges[str(abs(int(edge.get('id').split("#")[0])))] = []

    # when a car running on sumo simulation, there are some internal edges between edges
    # connections = {
    #     "internal_edge_id_1": (edge_id_in_sumo_1, edge_id_in_sumo_2),
    #     "internal_edge_id_2": (edge_id_in_sumo_3, edge_id_in_sumo_4),
    #     ...
    # }

    connections = {}
    for connection in root.findall('connection'):
        if connection.get('via'):
            connections[connection.get('via').rsplit('_', 1)[0]] = (connection.get('from'), connection.get('to'))

    # when we want to set the state of traffic light
    # we need to get the number of 'rgy' in state and fit it
    # tls = {
    #     "tl_id": number_of_'rgy'
    # }

    tls = {}
    for tl in root.findall('tlLogic'):
        tls[tl.get('id')] = len(tl.find('phase').get('state'))

    # complete the edges dict
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


# create the environment that rl model needs
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
