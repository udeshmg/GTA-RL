#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# Solve a traveling salesman problem on a set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import argparse
import numpy as np
from utils.data_utils import load_dataset, save_dataset
from gurobipy import *
import pickle
import time

def solve_euclidian_tap(points, adj, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    n = len(points)
    demand = points[-1][2]

    # Dictionary of Euclidean distance between each pair of points
    dist = {(i, j): math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i in range(n) for j in range(n)}

    m = Model()
    m.Params.outputFlag = False
    # Create variables

    vars = m.addVars(dist.keys(), vtype=GRB.INTEGER, name='e')

    obj = sum((dist[i,j]  + 0.2*(vars[i,j]+vars[j,i]))*(vars[i,j]+vars[j,i]) for i in range(n) for j in range(i))
    m.setObjective(obj, GRB.MINIMIZE)

    m.addConstrs(vars[i,j] == 0 for i in range(n) for j in range(n) if adj[i][j] == True)

    m.addConstr(sum(vars[0, i] for i in range(n) if i != 0) == demand)
    #m.addConstr(sum(vars[i, 0] for i in range(n)) == 0)
    m.addConstr(sum(vars[i, n-1] for i in range(n) if i != n-1) == demand)
    #m.addConstr(sum(vars[i, n - 1] for i in range(n)) == 0)

    m.addConstr(vars[n-1, 0] == 0)

    for i in range(1, n-1):
        m.addConstr(sum(vars[i,j] for j in range(n) if i != j) == sum(vars[j,i] for j in range(n) if i != j))

    # Optimize model

    m._vars = vars
    #m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage

    m.optimize()
    vals = m.getAttr('x', vars)

    return m.objVal, None

def solve_dynamic_euclidian_tap(points, adj, threads=0, timeout=None, gap=None):
    """
    Solves the Dynamic Euclidan TSP problem to optimality using the MIP formulation
    :param points: list of (t, x, y) coordinate
    :return:
    """

    n = len(points)
    demand = points[0][-1][2]

    # Dictionary of Euclidean distance between each pair of points


    dist = {}
    for i in range(n):
        for j in range(n):
            for t in range(n):
                nxt = 0 if (t == n-1) else t+1
                dist[t, i, j] = math.sqrt(sum((points[t][i][k] - points[nxt][j][k]) ** 2 for k in range(2)))

    m = Model()
    m.Params.outputFlag = False
    # Create variables

    vars = m.addVars(dist.keys(), vtype=GRB.INTEGER, name='e')

    for t in range(n):
        m.addConstrs(vars[t, i, j] == 0 for i in range(n) for j in range(n) if adj[i][j] == True)


    for t in range(n):
        if t == 0:
            m.addConstr(sum(vars[t, 0, i] for i in range(n) if i != 0) == demand)
        else:
            m.addConstr(sum(vars[t, 0, i] for i in range(n) if i != 0) == 0)
        m.addConstr(sum(vars[t, i, 0] for i in range(n) if i != 0) == 0)

    m.addConstr(sum(vars[t, i, n-1] for t in range(n) for i in range(n) if i != n-1) == demand)

    for i in range(0, n-1):
        m.addConstr(sum(vars[t, n-1, i] for t in range(n)) == 0)


    for i in range(1, n - 1):
        for t in range(n-1):
            nxt = t + 1
            m.addConstr(sum(vars[t, j, i] for j in range(n) if i != j) == sum(vars[nxt, i, j] for j in range(n) if i != j))

    #for t in range(n-1):
    #    nxt = t + 1
    #    m.addConstr(sum(vars[t, i, j] for i in range(n) for j in range(n) if i != j) ==
    #                sum(vars[nxt, i, j] for i in range(n) for j in range(n) if i != j))

    obj = sum((dist[t,i,j]  + 0.2*(vars[t, i, j]+vars[t, j, i]))*(vars[t, i, j]+vars[t, j, i])
              for t in range(n) for i in range(n) for j in range(i))

    m.setObjective(obj, GRB.MINIMIZE)
    # Optimize model
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize()

    vals = m.getAttr('x', vars)
    #selected = tuplelist((t, i, j) for t, i, j in vals.keys() if vals[t, i, j] > 0.5)

    # sort the selected nodes in time
    #tour = [i[1] for i in sorted(selected, key=lambda tup: tup[0])]

    #assert len(tour) == n

    return m.objVal, None



def solve_all_gurobi(dataset, dynamic, timeout):
    results = []
    start_time = time.time()
    i = 0
    for instance, adj in load_from_path(dataset):
        if dynamic:
            result, _ = solve_dynamic_euclidian_tap(instance, adj, timeout=timeout)
        else:
            result, _ = solve_euclidian_tap(instance, adj, timeout=timeout)
        print("Solved instance {} with tour length {} : Solved in {} seconds".format(i, result, time.time()-start_time))
        results.append(result)
        i += 1
    return sum(results)/i

def get(num_nodes, threshold):
    stack = []
    init = np.random.uniform(0, 1, (num_nodes, 2))
    for i in range(num_nodes):
        stack.append(init)
        init = np.clip(init + np.random.uniform(-threshold, threshold, (num_nodes, 2)), 0, 1)

    np_stack = np.array(stack)

    return np_stack

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def plot_tsp(xy, tour, ax1, total_xy=None):
    """
    Plot the TSP tour on matplotlib axis ax1.
    """

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)


    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()

    # Scatter nodes

    # Starting node
    ax1.scatter([xs[0]], [ys[0]], s=100, color='red')

    if total_xy is not None:
        time, nodes, coords = total_xy.shape
        flatten_xy =  total_xy.reshape((time*nodes, coords))
        colors = cm.rainbow(np.linspace(0, 1, nodes))

        for i in range(nodes):
            ax1.scatter(total_xy[:, i, 0], total_xy[:, i, 1], s=20, color=colors[i])
            ax1.scatter(xy[i][0], xy[i][1], s=100, color=colors[i])

    else:
        ax1.scatter(xs, ys, s=100, color='blue')

    # Arcs
    qv = ax1.quiver(
        xs, ys, dx, dy,
        scale_units='xy',
        angles='xy',
        scale=1,
    )

    ax1.set_title('{} nodes, total length {:.5f}'.format(len(tour), lengths[-1]))

def load_from_path(filename):

    assert os.path.splitext(filename)[1] == '.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

import torch
import random

def create_instance(size, demand_max):
    instance = torch.FloatTensor(size, 2).uniform_(0, 1)
    demand = random.randint(2, demand_max)
    demand_t = torch.zeros(size, 1)
    demand_t[0, 0] = -demand
    demand_t[-1, 0] = demand

    instance = torch.cat((instance, demand_t), dim=1)

    return instance.numpy()

def create_adj(size):
    adj = torch.zeros(size, size, dtype=torch.bool)
    coords = torch.randint(1, size-1, (size, size-5))
    adj = adj.scatter(1, coords, True)
    adj[size-1, 0] = False # there should always be a connection from destination to source for resetting the path
    return adj.numpy()

if __name__=="__main__":
    avg_cost = solve_all_gurobi("../../data/dynamic_tap/dynamic_tap10_threshold_0.5_seed4321.pkl", dynamic=True, timeout=None)
    print(avg_cost)
    """
    xy = create_instance(50, 20)
    adj = create_adj(50)
    #print(xy)
    tour_length, tour = solve_euclidian_tap(xy, adj)
    #tour_length, tour = solve_euclidian_tsp(xy)
    print("Tour length: ", tour_length, " Tour: ", tour)

    fig, ax = plt.subplots(figsize=(10, 10))

    ordered = np.array([np.arange(len(tour)), tour]).T
    ordered = ordered[ordered[:,1].argsort()]
    coords = xy[ordered[:,0], ordered[:,1]]
    plot_tsp(coords, tour, ax, xy)

    plt.show()
    """