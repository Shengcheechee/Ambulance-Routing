#!/usr/local/bin/python3
import networkx as nx
import random
from parse_osm import parser, render

def converter(nodes, ways):

    graph_nodes = []

    for node in nodes.keys():
        if len(nodes[node]['belonged_ways']) > 1:
            graph_nodes.append(node)

    osm_graph = nx.DiGraph()
    osm_graph.add_nodes_from(graph_nodes)
    for way in ways.keys():
        nd_tmp = ""
        for nd in ways[way][0]:
            if nd in graph_nodes:
                if ways[way][2] == 1:
                    if nd_tmp == "":
                        nd_tmp = nd
                    else:
                        osm_graph.add_edge(nd_tmp, nd, id = way)
                        nd_tmp = nd
                elif ways[way][2] == -1:
                    if nd_tmp == "":
                        nd_tmp = nd
                    else:
                        osm_graph.add_edge(nd, nd_tmp, id = way)
                        nd_tmp = nd
                else:
                    if nd_tmp == "":
                        nd_tmp = nd
                    else:
                        osm_graph.add_edge(nd_tmp, nd, id = way)
                        osm_graph.add_edge(nd, nd_tmp, id = way)
                        nd_tmp = nd

    return osm_graph

def shortest_path(nx_graph, s, t):

    sp = nx.shortest_path(nx_graph, source = s, target = t)

    path = []
    for i in range(len(sp)-1):
        path.append(nx_graph.get_edge_data(sp[i], sp[i+1])['id'])

    print("Source node id : ", sp[0])
    print("Target node id : ", sp[-1])
    print("Shortest path node ids: ", sp)
    print("Shortest path way ids : ", path)

    return sp, path

def get_subgraph(nx_graph, sp, radius):
    for i in range(radius):
        subgraph = []
        for point in sp:
            point_neighbors = list(nx_graph.neighbors(point))
            for j in range(len(point_neighbors)):
                if point_neighbors[j] not in subgraph:
                    subgraph.append(point_neighbors[j])
        print(subgraph)
        sp = subgraph

if __name__ == '__main__':
    nodes, ways = parser('ncku.osm')
    nx_graph = converter(nodes, ways)
    source = random.choice(list(nx_graph))
    target = random.choice(list(nx_graph))
    sp, path = shortest_path(nx_graph, source, target)
    render(nodes, ways, sp, path)
    get_subgraph(nx_graph, sp, 2)

