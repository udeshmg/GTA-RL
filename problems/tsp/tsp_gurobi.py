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

def solve_euclidian_tsp(points, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points
    dist = {(i, j): math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')

    for i, j in vars.keys():
        vars[j, i] = vars[i, j] # edge in opposite direction
    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))


    # Add degree-2 constraint

    #m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Using Python looping constructs, the preceding would be...
    #
    for i in range(n):
        m.addConstr(sum(vars[i,j] for j in range(n) if i != j) == 2)

    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n

    return m.objVal, tour

def solve_dynamic_euclidian_tsp(points, threads=0, timeout=None, gap=None):
    """
    Solves the Dynamic Euclidan TSP problem to optimality using the MIP formulation
    :param points: list of (t, x, y) coordinate
    :return:
    """

    n = len(points)

    # Dictionary of Euclidean distance between each pair of points


    dist = {}
    for i in range(n):
        for j in range(n):
            for t in range(n):
                nxt = 0 if (t == n-1) else t+1
                if i != j:
                    dist[t, i, j] = math.sqrt(sum((points[t][i][k] - points[nxt][j][k]) ** 2 for k in range(2)))

    m = Model()
    m.Params.outputFlag = False
    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')

    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)

    for i in range(n): # for a node there should 2 connections over the entire time horizon
        m.addConstr(sum(vars[t, i, j] + vars[t, j, i] for t in range(n) for j in range(n) if i != j ) == 2)

    for t in range(n): # At one time step there should be only one connection
        m.addConstr(sum(vars[t, i, j] for i in range(n) for j in range(n)  if i != j ) == 1)

    # If there is a connection between i,j at time t then
    # 1. i should be connected to some node at time t-1
    # 2. j should be connected to some node at time t+1
    for i in range(n):
        for j in range(n):
            if i != j:
                for t in range(n):
                    nxt = 0 if (t == n - 1) else t + 1
                    prev = n-1 if (t == 0) else t-1

                    m.addConstr(vars[t, i, j] <= sum(vars[nxt, j, k] for k in range(n) if k != j))
                    m.addConstr(vars[t, i, j] <= sum(vars[prev, k, i] for k in range(n) if k != i))

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
    selected = tuplelist((t, i, j) for t, i, j in vals.keys() if vals[t, i, j] > 0.5)

    # sort the selected nodes in time
    tour = [i[1] for i in sorted(selected, key=lambda tup: tup[0])]

    assert len(tour) == n

    return m.objVal, tour



def solve_all_gurobi(dataset, dynamic, timeout):
    results = []
    start_time = time.time()
    for i, instance in enumerate(dataset):
        if dynamic:
            result = solve_dynamic_euclidian_tsp(instance, timeout=timeout)
        else:
            result = solve_euclidian_tsp(instance, timeout=timeout)
        print("Solved instance {} with tour length {} : Solved in {} seconds".format(i,result, time.time()-start_time))
        results.append(result)
    return results

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

if __name__=="__main__":
    np.random.seed(1234)
    xy = np.random.uniform(0, 1, (20, 2))
    #xy = get(20, 0.1)
    #print(xy)
    tour_length, tour = solve_euclidian_tsp(xy)
    #tour_length, tour = solve_euclidian_tsp(xy)
    print("Tour length: ", tour_length, " Tour: ", tour)

    fig, ax = plt.subplots(figsize=(10, 10))

    ordered = np.array([np.arange(len(tour)), tour]).T
    ordered = ordered[ordered[:,1].argsort()]
    coords = xy[ordered[:,0], ordered[:,1]]
    plot_tsp(coords, tour, ax, xy)

    plt.show()