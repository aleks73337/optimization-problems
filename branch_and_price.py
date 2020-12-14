import os
import numpy as np
import cplex
import copy
from numpy.core.numeric import isclose
from tqdm import tqdm
import time
from func_timeout import func_timeout, FunctionTimedOut
import random
from branch_and_bound import Dataset
import networkx

import sys
sys.setrecursionlimit(1500)


class BranchAndPriceSolver:
    def __init__(self, dataset,  root_path):
        self.base_problem = cplex.Cplex()
        self.base_problem.set_results_stream(None)
        self.dataset = dataset
        self.colors = self.initialize_model()
    
    def improve_color(self, color_nodes):
        valid_nodes = [i for i in range(self.dataset.number_of_vertices)  if i not in color_nodes]
        get_independent_nodes = lambda  n: list(set([i for i in range(self.dataset.number_of_vertices) if (self.dataset.graph[i, n] == 0)]))
        for node in color_nodes:
            indep_nodes = get_independent_nodes(node)
            valid_nodes = [i for i in valid_nodes if i in indep_nodes]
        best_color = color_nodes
        while valid_nodes:
            node = valid_nodes.pop(0)
            best_color.append(node)
            indep_nodes = get_independent_nodes(node)
            valid_nodes = [i for i in valid_nodes if i in indep_nodes]
        return best_color
    
    def get_initial_colorings(self):
        graph = networkx.from_numpy_array(self.dataset.graph)
        nodes_colors = networkx.algorithms.coloring.greedy_color(graph)
        colors = {}
        for node, color in nodes_colors.items():
            cur_color = colors.get(color)
            if cur_color == None:
                colors[color] = [node]
            else:
                colors[color].append(node)
        res = []
        for color, nodes_set in colors.items():
            new_nodes_set = sorted(self.improve_color(nodes_set))
            if new_nodes_set not in res:
                res.append(new_nodes_set)
        return res
    
    def initialize_constrains(self, colors):
        for i in range(self.dataset.number_of_vertices):
            colors_with_node = []
            for color_idx, color_set in enumerate(colors):
                if i in color_set:
                    colors_with_node.append(color_idx)
            if len(colors_with_node) > 0:
                self.base_problem.linear_constraints.add(
                    cplex.SparsePair(ind = colors_with_node, val = [1.0] * len(colors_with_node)),
                    rhs = [1.0],
                    senses = ["G"]
                )
    
    def initialize_variables(self, colors):
        for idx, color_set in enumerate(colors):
            self.base_problem.variables.add(names = [f"y_{idx}"])
            self.base_problem.variables.set_lower_bounds(idx, 0.0)
            self.base_problem.variables.set_upper_bounds(idx, 1.0)
            self.base_problem.objective.set_linear(idx, 1.0)

    def initialize_model(self):
        colors = self.get_initial_colorings()
        self.initialize_variables(colors)
        self.initialize_constrains(colors)
        self.base_problem.objective.set_sense(self.base_problem.objective.sense.maximize)
        return colors

if __name__ == "__main__":
    root_path = os.path.dirname(__file__)
    data_folder = os.path.join(root_path, "data")
    data_paths = [os.path.join(data_folder, name) for name in os.listdir(data_folder)]

    descision_folder = os.path.join(root_path, "results")
    if not os.path.isdir(descision_folder):
        os.makedirs(descision_folder)
    
    # for idx, fname in enumerate(data_paths):
    #     print(f"{idx}. {fname}")
    # exit()

    global best_solution, best_score
    for path in data_paths[0:1]:
        graph_name = ''.join(path.split(os.sep)[-1].split('.')[0:-1])
        print(graph_name)
        res_path = os.path.join(descision_folder, graph_name + '.txt')
        dataset = Dataset(path)
        best_solution = 0
        best_score = 0
        solver = BranchAndPriceSolver(dataset, root_path)
        # start_time = time.time()
        # try:
        #     doitReturnValue = func_timeout(1800, solver, args=())
        # except Exception as e:
        #     print(e)
        # total_time = time.time() - start_time
        # print(total_time)

        # def save_results(descision, calc_time, n_vertices, path):
        #     with open(path, 'w+') as f:
        #         res_string = "N vertices: {} \nCalculation time: {} seconds \nDescision: {}".format(n_vertices, calc_time, descision)
        #         f.write(res_string)

        # try:
        #     save_results(best_solution, total_time, best_score, res_path)
        # except Exception as e:
        #     print(e)