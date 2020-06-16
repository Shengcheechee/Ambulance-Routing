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
    sp_nodes = nx.shortest_path(nx_graph, source = s, target = t)

    path_ids = []
    path_edges = []

    for i in range(len(sp)-1):
        path_ids.append(nx_graph.get_edge_data(sp[i], sp[i + 1])['id'])
        path_edges.append((sp[i], sp[i + 1]))

    print("Source node id : ", sp[0])
    print("Target node id : ", sp[-1])
    print("Shortest path node ids: ", sp)
    print("Shortest path way ids : ", path)

    return sp_nodes, path_ids, path_edges

def get_subgraph(nx_graph, edge):
    graph_1 = nx_graph.ego_graph(nx_graph, edge[0], radius = 5)
    graph_2 = nx_graph.ego_graph(nx_graph, edge[1], radius = 5)
    subgraph = nx.compose(graph_1, graph_2)

    return subgraph

if __name__ == '__main__':
    nodes, ways = parser('ncku.osm')
    nx_graph = converter(nodes, ways)
    source = random.choice(list(nx_graph))
    target = random.choice(list(nx_graph))
    sp_nodes, path_ids, path_edges = shortest_path(nx_graph, source, target)
    subgraphs = []
    for edge in edges:
        subgraphs.append(get_subgraph(nx_graph, edge))
    render(nodes, ways, sp_nodes, path_ids)
