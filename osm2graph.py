#!/usr/local/bin/python3
import networkx as nx
import random
from parse_osm import parser, render

def convert2graph(nodes, ways):
    osm_graph = nx.DiGraph()

    for node in nodes.keys():
        if len(nodes[node]['belonged_ways']) > 1:
            osm_graph.add_node(node, trafsig = nodes[node]['trafsig'])

    for way in ways.keys():
        nd_tmp = ""
        for nd in ways[way]['way_member']:
            if nd in list(osm_graph):
                if ways[way]['oneway']:
                    if nd_tmp == "":
                        nd_tmp = nd
                    else:
                        osm_graph.add_edge(nd_tmp, nd, id = way, weight = ways[way]['lane'] * ways[way]['priority'])
                        nd_tmp = nd
                else:
                    if nd_tmp == "":
                        nd_tmp = nd
                    else:
                        osm_graph.add_edge(nd_tmp, nd, id = way, weight = ways[way]['lane'] * ways[way]['priority'])
                        osm_graph.add_edge(nd, nd_tmp, id = way, weight = ways[way]['lane'] * ways[way]['priority'])
                        nd_tmp = nd

    return osm_graph

def shortest_path(nx_graph, src, tgt):
    sp_nodes = nx.shortest_path(nx_graph, source = src, target = tgt)

    path_ids = []
    for i in range(len(sp_nodes)-1):
        path_ids.append(nx_graph.get_edge_data(sp_nodes[i], sp_nodes[i + 1])['id'])

    # print("Source node id : ", sp_nodes[0])
    # print("Target node id : ", sp_nodes[-1])
    # print("Shortest path node ids: ", sp_nodes)

    return path_ids

def get_subgraph(nx_graph, n1, n2):
    graph_1 = nx.ego_graph(nx_graph, n1, radius = 2)
    graph_2 = nx.ego_graph(nx_graph, n2, radius = 2)
    subgraph = nx.compose(graph_1, graph_2)

    return subgraph

if __name__ == '__main__':
    nodes, ways = parser('ncku.osm')
    nx_graph = convert2graph(nodes, ways)
    source = "5520434289"
    target = "355961064"
    path_ids = shortest_path(nx_graph, source, target)
