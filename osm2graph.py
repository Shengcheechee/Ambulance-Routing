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
        for nd in ways[way][0]:
            if nd in list(osm_graph):
                if ways[way][2] == 1:
                    if nd_tmp == "":
                        nd_tmp = nd
                    else:
                        osm_graph.add_edge(nd_tmp, nd, id = way, lane = ways[way][1], priority = ways[way][3])
                        nd_tmp = nd
                elif ways[way][2] == -1:
                    if nd_tmp == "":
                        nd_tmp = nd
                    else:
                        osm_graph.add_edge(nd, nd_tmp, id = way, lane = ways[way][1], priority = ways[way][3])
                        nd_tmp = nd
                else:
                    if nd_tmp == "":
                        nd_tmp = nd
                    else:
                        osm_graph.add_edge(nd_tmp, nd, id = way, lane = ways[way][1], priority = ways[way][3])
                        osm_graph.add_edge(nd, nd_tmp, id = way, lane = ways[way][1], priority = ways[way][3])
                        nd_tmp = nd

    return osm_graph

def shortest_path(nx_graph, s, t):
    sp_nodes = nx.shortest_path(nx_graph, source = s, target = t)

    path_ids = []
    path_edges = []

    for i in range(len(sp_nodes)-1):
        path_ids.append(nx_graph.get_edge_data(sp_nodes[i], sp_nodes[i + 1])['id'])
        path_edges.append((sp_nodes[i], sp_nodes[i + 1]))

    print("Source node id : ", sp_nodes[0])
    print("Target node id : ", sp_nodes[-1])
    print("Shortest path node ids: ", sp_nodes)
    print("Shortest path way ids : ", path_ids)

    return sp_nodes, path_ids, path_edges

def get_subgraph(nx_graph, edge):
    graph_1 = nx.ego_graph(nx_graph, edge[0], radius = 2)
    graph_2 = nx.ego_graph(nx_graph, edge[1], radius = 2)
    subgraph = nx.compose(graph_1, graph_2)

    return subgraph

if __name__ == '__main__':
    nodes, ways = parser('ncku.osm')
    nx_graph = convert2graph(nodes, ways)
    source = random.choice(list(nx_graph))
    target = random.choice(list(nx_graph))
    sp_nodes, path_ids, path_edges = shortest_path(nx_graph, source, target)
    subgraphs = []
    for edge in path_edges:
        subgraphs.append(get_subgraph(nx_graph, edge))
    x_min, y_max = render(nodes, ways, sp_nodes, path_ids, subgraphs)
