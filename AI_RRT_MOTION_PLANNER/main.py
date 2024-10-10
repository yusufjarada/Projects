from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
import math
import os
import random
import sys
import heapq
from model import NeuralNet
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from collision_utils import get_collision_fn

UR5_JOINT_INDICES = [0, 1, 2]


def dist(Node1_conf, Node2_conf):
    distance = np.linalg.norm(np.subtract(Node1_conf, Node2_conf))
    return distance


def sample_conf():
    theta1 = random.uniform(-2 * math.pi, 2 * math.pi)
    theta2 = random.uniform(-2 * math.pi, 2 * math.pi)
    theta3 = random.uniform(-1 * math.pi, 1 * math.pi)

    sampled_conf = (theta1, theta2, theta3)

    while collision_fn(sampled_conf):
        theta1 = random.uniform(-2 * math.pi, 2 * math.pi)
        theta2 = random.uniform(-2 * math.pi, 2 * math.pi)
        theta3 = random.uniform(-1 * math.pi, 1 * math.pi)

        sampled_conf = (theta1, theta2, theta3)

    return sampled_conf


class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_vertex(self, vertex):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = {}

    def add_edge(self, vertex1, vertex2, weight):
        if vertex1 in self.adjacency_list and vertex2 in self.adjacency_list:
            self.adjacency_list[vertex1][vertex2] = weight
            self.adjacency_list[vertex2][vertex1] = weight  # Uncomment this line for undirected graph

    def remove_vertex(self, vertex):
        if vertex in self.adjacency_list:
            del self.adjacency_list[vertex]
            for vertices in self.adjacency_list.values():
                if vertex in vertices:
                    del vertices[vertex]

    def remove_edge(self, vertex1, vertex2):
        if vertex1 in self.adjacency_list and vertex2 in self.adjacency_list:
            if vertex2 in self.adjacency_list[vertex1]:
                del self.adjacency_list[vertex1][vertex2]
            if vertex1 in self.adjacency_list[vertex2]:
                del self.adjacency_list[vertex2][vertex1]

    def get_neighbors(self, vertex):
        if vertex in self.adjacency_list:
            return self.adjacency_list[vertex]
        else:
            return {}  # change this if needed for the checking of neighbors

    def areNeighbors(self, vertex1, vertex2):
        # Returns true if the vertexs have each other in their neighbors list
        # else returns true
        pass

    def __str__(self):
        result = ""
        for vertex, neighbors in self.adjacency_list.items():
            result += str(vertex) + " -> " + str(neighbors) + "\n"
        return result


class RRT_Node:
    def __init__(self, conf):
        self.conf = conf
        self.parent = None
        self.children = []
        self.cost = 0
        self.HasBeenRewired = False
        self.inTree = False
    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)


def steer_to(sample, neighbor):
    tinyVector = tuple(np.subtract(neighbor, sample) / 100)
    for i in range(100):
        if collision_fn(sample):
            return False
        sample = tuple(x + y for x, y in zip(sample, tinyVector))
    return True


def steer_to_with_time(sample, neighbor):
    tinyVector = tuple(np.subtract(neighbor, sample) / 100)
    for i in range(100):

        if collision_fn(sample):
            return False
        sample = tuple(x + y for x, y in zip(sample, tinyVector))
        time.sleep(0.01)
    return True


def steer_to_with_time_and_draw(sample, neighbor,color):
    tinyVector = tuple(np.subtract(neighbor, sample) / 100)
    for i in range(100):
        location = get_end_effector_position(ur5, UR5_JOINT_INDICES, sample)
        if(i % 5 == 0):
            goal_marker = draw_sphere_marker(position=location, radius=0.02, color=color)
        if collision_fn(sample):
            return False
        sample = tuple(x + y for x, y in zip(sample, tinyVector))
        time.sleep(0.01)
    return True


def getKnearestNodes(target, roadmap):
    distances = [(vertex, dist(target, vertex)) for vertex in roadmap.adjacency_list.keys() if vertex != target]
    distances.sort(key=lambda x: x[1])  # Sort by distance
    return distances[:3]


def find_nearest(rand_node, node_list):
    nearestNode = node_list[0]
    smallestDist = dist(rand_node.conf, nearestNode.conf)
    for node in node_list:
        currentDist = dist(rand_node.conf, node.conf)
        if currentDist < smallestDist:
            smallestDist = currentDist
            nearestNode = node

    return nearestNode


def addStartandGoal(graph, start_conf, goal_conf):
    Startflag = False  # this is to check if the start and goal were added
    Goalflag = False

    graph.add_vertex(start_conf)
    KnearestNodes = getKnearestNodes(start_conf, graph)
    for node in KnearestNodes:
        nodeCoords = node[0]
        distance = node[1]
        if steer_to(start_conf, nodeCoords):
            graph.add_edge(start_conf, nodeCoords, distance)
            Startflag = True

    graph.add_vertex(goal_conf)
    KnearestNodes = getKnearestNodes(goal_conf, graph)
    for node in KnearestNodes:
        nodeCoords = node[0]
        distance = node[1]
        if steer_to(goal_conf, nodeCoords):
            graph.add_edge(goal_conf, nodeCoords, distance)
            Goalflag = True

    if Startflag and Goalflag:
        return graph
    return []  # Return nothing if start and goal can not be added


def getPRMGraph():
    graph = Graph()

    for i in range(1000):
        print(i)
        sample = sample_conf()
        graph.add_vertex(sample)
        KnearestNodes = getKnearestNodes(sample, graph)
        for node in KnearestNodes:
            nodeCoords = node[0]
            distance = node[1]
            if steer_to(sample, nodeCoords):  # does not check if sample already exists in graph because the chances
                # of that happening are extremely low
                graph.add_edge(sample, nodeCoords, distance)

    return graph


def GetNearNodes(tree, SampledNode):
    numberOfNodes = len(tree) + 1
    radius = 5 * (np.log(numberOfNodes) / numberOfNodes) ** (1 / 3)  # following equation from notes

    listOfNearNodes = []

    for node in tree:

        if dist(node.conf, SampledNode.conf) <= radius:
            if steer_to(node.conf, SampledNode.conf):
                listOfNearNodes.append(node)
    return listOfNearNodes


def getBestParentNode(SampledNode, QNodes):
    if len(QNodes) == 0:
        return None

    bestParent = None
    cheapestCost = float('inf')
    for node in QNodes:
        if steer_to(SampledNode.conf, node.conf):
            SampledNode_cost = dist(node.conf, SampledNode.conf)
            if node.cost + SampledNode_cost < cheapestCost:
                cheapestCost = node.cost + SampledNode_cost
                bestParent = node
                SampledNode.cost = SampledNode_cost

    return bestParent


def RRT_Star(start_conf, goal_conf):
    startNode = RRT_Node(start_conf)
    startNode.cost = get_Heurisitc_from_Model(start_conf,goal_conf,startNode.conf)
    tree = [startNode]
    goalFound = False
    BestPathLength = float('inf')
    for i in range(0, 500):
        # print(i)
        SampledNode = RRT_Node(sample_conf())
        NearestNode = find_nearest(SampledNode, tree)

        if steer_to(SampledNode.conf, NearestNode.conf):

            QNodes = GetNearNodes(tree, SampledNode)
            BestParentNode = getBestParentNode(SampledNode, QNodes)
            if BestParentNode is not None:
                SampledNode.parent = BestParentNode
                BestParentNode.add_child(SampledNode)
                SampledNode.heuristic = get_Heurisitc_from_Model(start_conf, goal_conf, SampledNode.conf)

                SampledNode.cost = BestParentNode.cost + dist(SampledNode.conf, BestParentNode.conf) + SampledNode.heuristic
                # SampledNode.cost = BestParentNode.cost + dist(SampledNode.conf, BestParentNode.conf)
                tree.append(SampledNode)
                SampledNode.inTree = True
                QNodes.append(SampledNode)
                rewire(SampledNode, QNodes, start_conf, goal_conf)

            if steer_to(SampledNode.conf, goal_conf) and dist(SampledNode.conf, goal_conf) < 2:
                if SampledNode.inTree:
                    goalFound = True
                    path = getPath(SampledNode, goal_conf)
                    pathLength = getPathLength(path)
                    if BestPathLength > pathLength and path is not [] and len(path) > 2:
                        finalPath = path
                        BestPathLength = pathLength
                        drawpath(finalPath)
                        print("Length of the Path:", BestPathLength)


    if not goalFound:
        print("goal did not attach to tree, may need more points")
        return None

    return finalPath


def getPath(node, goal_conf):
    finalPath = []
    currentNode = node
    loops = 0
    while currentNode is not None:
        if loops > 100:
            return []
        finalPath.append(currentNode.conf)
        currentNode = currentNode.parent
        loops += 1
    finalPath = finalPath[::-1]
    finalPath.append(goal_conf)
    return finalPath

def rewireChildren(Node, deltaCost,depth):
    if(depth > 5):
        return None

    for child in Node.children:
        rewireChildren(child,deltaCost,depth+1)
        child.cost -= deltaCost
def rewire(SampledNode, QNodes, start_conf, goal_conf):
    for node in QNodes:
        if not node.HasBeenRewired:
            newCost = SampledNode.cost + dist(SampledNode.conf, node.conf)
            if node.cost > newCost:
                deltaCost = node.cost - newCost
                rewireChildren(node, deltaCost,0)
                node.cost = newCost
                node.parent = SampledNode
                SampledNode.add_child(node)


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def dijkstra(graph, source):
    distances = {vertex: float('inf') for vertex in graph.adjacency_list}
    distances[source] = 0
    queue = [(0, source)]  # Priority queue to store nodes with their distances

    while queue:
        current_distance, current_vertex = heapq.heappop(queue)

        # If the current distance is greater than the recorded distance, skip
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph.adjacency_list[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                # Remove the outdated entry from the priority queue
                queue = [(d, v) for d, v in queue if v != neighbor]
                # Add the updated distance and neighbor back into the priority queue
                heapq.heappush(queue, (distance, neighbor))

    return distances


def shortest_path(graph, source, target):
    if not graph:
        return []
    graph = addStartandGoal(graph, start_conf=source, goal_conf=target)
    if graph == []:
        print("Could not add start and goal, may require more nodes")
        return []
    distances = dijkstra(graph, source)
    path = [target]
    visited = set()
    cycles = 0
    while target != source:
        cycles += 1
        visited.add(target)
        if len(visited) == len(graph.adjacency_list):
            print("Got stuck in a loop")
            return []
        for neighbor, weight in graph.adjacency_list[target].items():
            if distances[target] == distances[neighbor] + weight and neighbor not in visited:
                path.append(neighbor)
                target = neighbor
                break
        if (cycles > 1000):
            return []
    return list(reversed(path))


def getCostOfPoint(path, point):
    startIndex = path.index(point)
    totalCost = 0
    for index in range(startIndex, len(path) - 1):
        totalCost += dist(path[index], path[index + 1])
    return totalCost


def extract_waypoints_and_cost_to_goal(path):
    if path is []:
        return None
    waypoints = {}
    for point in path:
        cost = getCostOfPoint(path, point)
        waypoints[point] = cost
    return waypoints


def draw_sphere_marker(position, radius, color):
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
    return marker_id


def remove_marker(marker_id):
    p.removeBody(marker_id)


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--a', action='store_true', default=False)
    args = parser.parse_args()
    return args


def getPathLength(path):
    total = 0
    for i in range(len(path) - 1):
        total += dist(path[i], path[i + 1])
    return total

def drawpath(path_conf):
    if path_conf is not None:
        pathDrawn = False
        for i in range(1):
            for index, q in enumerate(path_conf):
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                if not pathDrawn:
                    steer_to_with_time_and_draw(q, path_conf[index + 1],[0, 1, 0, 1])
                else:
                    steer_to_with_time(q, path_conf[index + 1])
                time.sleep(.1)
                if (index == len(path_conf) - 2):
                    pathDrawn = True
                    break


def drawpath_final(path_conf):
    if path_conf is not None:
        pathDrawn = False
        for i in range(1):
            for index, q in enumerate(path_conf):
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                if not pathDrawn:
                    steer_to_with_time_and_draw(q, path_conf[index + 1],[1, 1, 0, 1])
                else:
                    steer_to_with_time(q, path_conf[index + 1])
                time.sleep(.1)
                if (index == len(path_conf) - 2):
                    pathDrawn = True
                    break

def load_model(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Load more keys if needed
    model.eval()  # Set model to evaluation mode
    return model

def get_end_effector_position(ur5, joint_indices, configuration):
    # Set robot joints to the given configuration
    set_joint_positions(ur5, joint_indices, configuration)

    # Get the transformation matrix of the end effector
    link_state = p.getLinkState(ur5, 3, computeForwardKinematics=True)  # Assuming link index 7 is the end effector
    end_effector_pos = link_state[0]  # Position of the end effector in world coordinates
    return end_effector_pos

def get_Heurisitc_from_Model(start,goal,configuration):
    v1, v2, v3 = start
    v4, v5, v6 = goal
    v7, v8, v9 = configuration

    tensor = torch.tensor([v1, v2, v3, v4, v5, v6, v7, v8, v9],
                          dtype=torch.float32)

    tensor = tensor.to(device)

    output = model(tensor)
    return output.item()



if __name__ == "__main__":

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200,
                                 cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)

    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[2 / 4, 1 / 3, 1 / 2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2 / 4, 1 / 3, 1 / 6],
                           useFixedBase=True)
    obstacle3 = p.loadURDF('assets/block.urdf',
                           basePosition=[-1 / 3, 1 / 3, 1 / 3],
                           useFixedBase=True)
    obstacle4 = p.loadURDF('assets/block.urdf',
                           basePosition=[1 / 4, -1 / 2, 1 / 3],
                           useFixedBase=True)
    obstacle5 = p.loadURDF('assets/block.urdf',
                           basePosition=[1 / 4, 1 / 4, 1 / 3],
                           useFixedBase=True)

    obstacles = [plane, obstacle1, obstacle2,obstacle3,obstacle4,obstacle5]

    # start and goal
    start_configurations = [(np.pi / 2, -np.pi / 4, np.pi / 2), (0, -np.pi / 2, 0), (np.pi / 4, -np.pi / 2, -np.pi / 2)]
    goal_configurations = [(0, 0, 0), (-np.pi / 2, -np.pi / 4, np.pi / 2), (0, -np.pi / 2, 0)]

    goal_position = (5 / 6, 0, 1 / 8)

    start_position = get_end_effector_position(ur5, UR5_JOINT_INDICES, start_configurations[0])

    start_marker = draw_sphere_marker(position=start_position, radius=0.05, color=[1, 0, 1, 1])

    goal_marker = draw_sphere_marker(position=goal_position, radius=0.05, color=[1, 1, 0, 1])

    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                    attachments=[], self_collisions=True,
                                    disabled_collisions=set())



    # Setup Model
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    input_size = 9  # Define based on the number of features
    hidden_size1 = 64
    hidden_size2 = 32
    hidden_size3 = 16
    output_size = 1
    model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = load_model(model, optimizer, 'model_checkpoint.pth')
    model = model.to(device)


    # Example
    # output = get_Heurisitc_from_Model(start_configurations[0],goal_configurations[0],goal_configurations[0])


    path_conf = RRT_Star(start_configurations[0], goal_configurations[0])

    if path_conf is not None:
        for i in range(3):
            drawpath_final(path_conf)
            print("FINAL PATH COST:",getPathLength(path_conf))
    exit()


    # Set up PRM for data: DONE

    # PRMGraph = getPRMGraph()
    #
    # output_file_path = "data/env1"
    # with open(output_file_path, 'w') as output_file:
    #     # for start_conf in start_configurations:
    #     #     for goal_conf in goal_configurations:
    #     for i in range(10000):
    #         print(i)
    #         start_conf = sample_conf()
    #         goal_conf = sample_conf()
    #
    #         path_conf = shortest_path(PRMGraph, start_conf, goal_conf)
    #
    #         waypoints = extract_waypoints_and_cost_to_goal(path_conf)
    #
    #         if len(waypoints) > 0:
    #             output_file.write("Start Configuration: {}\n".format(start_conf))
    #             output_file.write("Goal Configuration: {}\n".format(goal_conf))
    #             output_file.write("Waypoints and Their Costs:\n")
    #             for point, cost in waypoints.items():
    #                 output_file.write("Waypoint: {} - Cost: {}\n".format(point, cost))
    #             output_file.write("\n")  # Separate configurations with a blank line
    #         PRMGraph.remove_vertex(start_conf)
    #         PRMGraph.remove_vertex(goal_conf)
    #         waypoints = None