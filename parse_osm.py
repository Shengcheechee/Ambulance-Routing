#!/usr/local/bin/python3
import xml.etree.ElementTree as ET
import numpy as np
import cv2

def parse_osm(filename):

    tree = ET.parse(filename)
    root = tree.getroot()

    node_info = {}

    for node in root.findall('node'):

        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        lat = int(lat * 100000 + 0.5)
        lon = int(lon * 100000 + 0.5)

        trafsig = False
        if node.find('tag') != None:
            if node.find('tag').get('v') == "traffic_signals":
                trafsig = True

        node_info[node.get('id')] = (lon, lat, trafsig)

    ways = []

    for way in root.findall('way'):

        way_mem = []
        lane = 1
        oneway = 0

        for tag in way.findall('tag'):

            if (tag.get('k') == "highway" and tag.get('v') not in {"path", "footway", "steps", "pedestrian"}):
                for nd in way.findall('nd'):
                    way_mem.append(nd.get('ref'))

            if tag.get('k') == "lanes":
                lane = int(tag.get('v'))

            if tag.get('k') == "oneway":
                if tag.get('v') == "yes":
                    oneway = 1
                if tag.get('v') == "-1":
                    oneway = -1

        if way_mem:
            ways.append((way_mem, lane, oneway))

    lon_min_id = min(node_info.keys(), key = lambda nid: node_info[nid][0])
    lon_max_id = max(node_info.keys(), key = lambda nid: node_info[nid][0])
    lat_min_id = min(node_info.keys(), key = lambda nid: node_info[nid][1])
    lat_max_id = max(node_info.keys(), key = lambda nid: node_info[nid][1])
    x_min = node_info[lon_min_id][0]
    x_max = node_info[lon_max_id][0]
    y_min = node_info[lat_min_id][1]
    y_max = node_info[lat_max_id][1]

    width = x_max - x_min + 40
    height = y_max - y_min + 40

    img = np.zeros((height, width, 3), np.uint16)
    img.fill(250)

    for way in ways:

        nd_temp = ""

        for nd in way[0]:

            if nd_temp == "":
                nd_temp = nd

            else:
                if way[2] == 1:
                    cv2.line(img, (node_info[nd_temp][0] - x_min, y_max - node_info[nd_temp][1]), (node_info[nd][0] - x_min, y_max - node_info[nd][1]), (200, 0, 0), way[1])
                elif way[2] == -1:
                    cv2.line(img, (node_info[nd_temp][0] - x_min, y_max - node_info[nd_temp][1]), (node_info[nd][0] - x_min, y_max - node_info[nd][1]), (0, 200, 0), way[1])
                else:
                    cv2.line(img, (node_info[nd_temp][0] - x_min, y_max - node_info[nd_temp][1]), (node_info[nd][0] - x_min, y_max - node_info[nd][1]), (150, 150, 150), way[1])
                nd_temp = nd

            cv2.circle(img, (node_info[nd][0] - x_min, y_max - node_info[nd][1]), 2, (0, 0, 0), -1)

            if node_info[nd][2] == True:
                cv2.circle(img, (node_info[nd][0] - x_min, y_max - node_info[nd][1]), 4, (0, 0, 255), 2)

            # cv2.putText(img, nd, (node_info[nd][0] - x_min + 1, y_max - node_info[nd][1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 128, 128), 1, cv2.LINE_AA)

    cv2.imwrite('My_image_3.jpg', img)

if __name__ == '__main__':
    parse_osm('nn.osm')
