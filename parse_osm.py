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

        trafsig = False
        if node.find('tag') != None:
            if node.find('tag').get('v') == "traffic_signals":
                trafsig = True

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
        oneway = 0

        for tag in way.findall('tag'):

            if (tag.get('k') == "highway" and tag.get('v') not in {"path", "footway", "steps", "pedestrian"}):
                for nd in way.findall('nd'):
                    way_mem.append(nd.get('ref'))
                    nodes[nd.get('ref')]['belonged_ways'].add(way.get('id'))

            if tag.get('k') == "lanes":
                lane = int(tag.get('v'))

            if tag.get('k') == "oneway":
                if tag.get('v') == "yes":
                    oneway = 1
                if tag.get('v') == "-1":
                    oneway = -1

        if way_mem:
            ways[way.get('id')] = (way_mem, lane, oneway)

    return nodes, ways

def render(nodes, ways, sp, path):

    lon_min_id = min(nodes.keys(), key = lambda nid: nodes[nid]['position'][0])
    lon_max_id = max(nodes.keys(), key = lambda nid: nodes[nid]['position'][0])
    lat_min_id = min(nodes.keys(), key = lambda nid: nodes[nid]['position'][1])
    lat_max_id = max(nodes.keys(), key = lambda nid: nodes[nid]['position'][1])
    x_min = nodes[lon_min_id]['position'][0]
    x_max = nodes[lon_max_id]['position'][0]
    y_min = nodes[lat_min_id]['position'][1]
    y_max = nodes[lat_max_id]['position'][1]

    width = x_max - x_min + 40
    height = y_max - y_min + 40

    img = np.zeros((height, width, 3), np.uint16)
    img.fill(250)

    for way in ways.keys():

        nd_tmp = ""
        start = False

        for nd in ways[way][0]:

            if nd_tmp == "":
                nd_tmp = nd

            else:
                if ways[way][2] == 1:
                    cv2.line(img, (nodes[nd_tmp]['position'][0] - x_min, y_max - nodes[nd_tmp]['position'][1]), (nodes[nd]['position'][0] - x_min, y_max - nodes[nd]['position'][1]), (200, 0, 0), ways[way][1])
                elif ways[way][2] == -1:
                    cv2.line(img, (nodes[nd_tmp]['position'][0] - x_min, y_max - nodes[nd_tmp]['position'][1]), (nodes[nd]['position'][0] - x_min, y_max - nodes[nd]['position'][1]), (0, 200, 0), ways[way][1])
                else:
                    cv2.line(img, (nodes[nd_tmp]['position'][0] - x_min, y_max - nodes[nd_tmp]['position'][1]), (nodes[nd]['position'][0] - x_min, y_max - nodes[nd]['position'][1]), (150, 150, 150), ways[way][1])
                nd_tmp = nd

            cv2.circle(img, (nodes[nd]['position'][0] - x_min, y_max - nodes[nd]['position'][1]), 2, (0, 0, 0), -1)

            if nodes[nd]['trafsig'] == True:
                cv2.circle(img, (nodes[nd]['position'][0] - x_min, y_max - nodes[nd]['position'][1]), 4, (0, 0, 255), 2)

            if len(nodes[nd]['belonged_ways']) > 1:
                if nd in sp:
                    cv2.circle(img, (nodes[nd]['position'][0] - x_min, y_max - nodes[nd]['position'][1]), 6, (0, 200, 200), 2)
                    cv2.putText(img, nd, (nodes[nd]['position'][0] - x_min + 1, y_max - nodes[nd]['position'][1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 128, 128), 1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (nodes[nd]['position'][0] - x_min, y_max - nodes[nd]['position'][1]), 6, (200, 200, 0), 2)


    for i in range(len(path)):

        shortest_path = []

        if ways[path[i]][0].index(sp[i + 1]) > ways[path[i]][0].index(sp[i]):
            shortest_path = ways[path[i]][0][ways[path[i]][0].index(sp[i]) : ways[path[i]][0].index(sp[i + 1]) + 1]
        else:
            shortest_path = ways[path[i]][0][ways[path[i]][0].index(sp[i + 1]) : ways[path[i]][0].index(sp[i]) + 1]

        sh_pa_tmp = ""

        for sh_pa in shortest_path:
            if sh_pa_tmp == "":
                sh_pa_tmp = sh_pa
            else:
                cv2.line(img, (nodes[sh_pa_tmp]['position'][0] - x_min, y_max - nodes[sh_pa_tmp]['position'][1]), (nodes[sh_pa]['position'][0] - x_min, y_max - nodes[sh_pa]['position'][1]), (0, 0, 200), ways[path[i]][1])
                sh_pa_tmp = sh_pa

    cv2.imwrite('My_map.jpg', img)

if __name__ == '__main__':
    nodes, ways = parser('ncku.osm')
    render(nodes, ways, [], [])
