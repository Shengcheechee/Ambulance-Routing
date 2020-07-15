#!/usr/local/bin/python3
import xml.etree.ElementTree as ET
import numpy as np
import cv2

def parser(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    nodes = {}

    for node in root.findall('node'):
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        lat = int(lat * 100000 + 0.5)
        lon = int(lon * 100000 + 0.5)
        trafsig = node.find('tag') != None and node.find('tag').get('v') == "traffic_signals"
        node_info = {
            'position': (lon, lat),
            'trafsig': trafsig,
            'belonged_ways': set()
        }
        nodes[node.get('id')] = node_info

    ways = {}

    for way in root.findall('way'):
        way_mem = []
        lane = 1
        priority = ""
        way_name = ""

        for tag in way.findall('tag'):
            if (tag.get('k') == "highway" and tag.get('v') not in {"path", "footway", "steps", "pedestrian"}):
                if tag.get('v') == "primary":
                    priority = 10
                elif tag.get('v') == "secondary":
                    priority = 8
                elif tag.get('v') == "tertiary":
                    priority = 6
                else:
                    priority = 1

                for nd in way.findall('nd'):
                    way_mem.append(nd.get('ref'))
                    nodes[nd.get('ref')]['belonged_ways'].add(way.get('id'))

            if tag.get('k') == "lanes":
                lane = int(tag.get('v'))

            if tag.get('k') == "oneway":
                oneway =  tag.get('v') == "yes" or tag.get('v') == "-1"
                if tag.get('v') == "-1":
                    way_mem.reverse()
            else:
                oneway = False

        if way_mem:
            ways[way.get('id')] = {
                'way_member': way_mem,
                'lane': lane,
                'oneway': oneway,
                'priority': priority
            }

    return nodes, ways

def gen_affine(x_min, y_max):
    def affine(node):
        return (node['position'][0] - x_min, y_max - node['position'][1])
    return affine

def render(nodes, ways, sp_nodes, path_ids, subgraphs):
    lon_min_id = min(nodes.keys(), key = lambda nid: nodes[nid]['position'][0])
    lon_max_id = max(nodes.keys(), key = lambda nid: nodes[nid]['position'][0])
    lat_min_id = min(nodes.keys(), key = lambda nid: nodes[nid]['position'][1])
    lat_max_id = max(nodes.keys(), key = lambda nid: nodes[nid]['position'][1])
    x_min = nodes[lon_min_id]['position'][0]
    x_max = nodes[lon_max_id]['position'][0]
    y_min = nodes[lat_min_id]['position'][1]
    y_max = nodes[lat_max_id]['position'][1]

    affine = gen_affine(x_min, y_max)

    width = x_max - x_min + 40
    height = y_max - y_min + 40

    img = np.zeros((height, width, 3), np.uint16)
    img.fill(250)

    for way in ways.keys():
        nd_tmp = ""
        start = False

        for nd in ways[way]['way_member']:
            if nd_tmp == "":
                nd_tmp = nd

            else:
                if ways[way]['oneway']:
                    cv2.line(img, affine(nodes[nd_tmp]), affine(nodes[nd]), (200,   0,   0), ways[way]['lane'])
                else:
                    cv2.line(img, affine(nodes[nd_tmp]), affine(nodes[nd]), (150, 150, 150), ways[way]['lane'])
                nd_tmp = nd

            cv2.circle(img, affine(nodes[nd]), 2, (0, 0, 0), -1)

            if nodes[nd]['trafsig']:
                cv2.circle(img, affine(nodes[nd]), 4, (0, 0, 255), 2)

            if len(nodes[nd]['belonged_ways']) > 1:
                if nd in sp_nodes:
                    cv2.circle(img, affine(nodes[nd]), 6, (0, 200, 200), 2)
                    cv2.putText(img, nd, (nodes[nd]['position'][0] - x_min + 1, y_max - nodes[nd]['position'][1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 128, 128), 1, cv2.LINE_AA)

                else:
                    cv2.circle(img, affine(nodes[nd]), 6, (200, 200, 0), 2)

    for i in range(len(path_ids)):
        shortest_path_ids = ways[path_ids[i]]['way_member'][ways[path_ids[i]]['way_member'].index(sp_nodes[i]) : ways[path_ids[i]]['way_member'].index(sp_nodes[i + 1]) + 1]
        for j in range(len(shortest_path_ids) - 1):
            cv2.line(img, affine(nodes[shortest_path_ids[j]]), affine(nodes[shortest_path_ids[j + 1]]), (0, 0, 200), ways[path_ids[i]]['lane'])

    count = 0
    for subgraph in subgraphs:
        for sg_nd in list(subgraph):
            cv2.circle(img, (nodes[sg_nd]['position'][0] - x_min, y_max - nodes[sg_nd]['position'][1]), 10, (200, 0, 200), 2)
            cv2.putText(img, sg_nd, (nodes[sg_nd]['position'][0] - x_min + 1, y_max - nodes[sg_nd]['position'][1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 128, 128), 1, cv2.LINE_AA)
        cv2.imwrite('My_map_%d.jpg'%(count), img)
        count += 1

    return x_min, y_max

if __name__ == '__main__':
    nodes, ways = parser('ncku.osm')
    render(nodes, ways, [], [], [])
