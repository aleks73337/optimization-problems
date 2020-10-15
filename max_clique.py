import os
import numpy as np
import cplex
import networkx
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, dataset_path):
        self.number_of_vertices = None
        self.graph = None
        self.parse_dataset(self.read_dataset(dataset_path))
    
    def read_dataset(self, data_path):
        with open(data_path, 'r') as f:
            data = f.readlines()
        return data

    def parse_dataset(self, input):
        for line in input:
            if (line[0] == 'c'):
                if ('number of vertices' in line):
                    self.number_of_vertices = int(line.split(' ')[5])
                    break
        self.graph = np.zeros(shape = [self.number_of_vertices, self.number_of_vertices])
        for line in input:
            line = line.replace('\n', '')
            if (line[0] == 'e'):
                _, vertex_from, vertex_to = line.split(' ')
                vertex_from = int(vertex_from) - 1
                vertex_to = int(vertex_to) - 1
                self.graph[vertex_from, vertex_to] = 2
                self.graph[vertex_to, vertex_from] = 2
        # self.graph = self.graph.tolist()


class MaxCliqueProblemSolver():
    def __init__(self, dataset : Dataset, root_path : str):
        self.problem = cplex.Cplex()
        self.variables = ['y' + str(i) for i in range(dataset.number_of_vertices)]
        self.problem.variables.add(names = self.variables, types = [self.problem.variables.type.integer] * dataset.number_of_vertices)

        for i, variable in enumerate(self.variables):
            self.problem.variables.set_lower_bounds(i, 0.0)
            self.problem.variables.set_upper_bounds(i, 1.0)

        for i in range(dataset.number_of_vertices):
            for j in range(dataset.number_of_vertices):
                if ((dataset.graph[i, j] == 0) & (i != j)):
                    self.problem.linear_constraints.add(
                        lin_expr = [cplex.SparsePair(ind = [i, j], val = [1.0, 1.0])],
                        rhs = [1.0],
                        names = ["y_" + str(i) + "_" + str(j)],
                        senses = ["L"]
                    )

        for variable in self.variables:
                self.problem.objective.set_linear([(variable, 1.0)])
        self.problem.objective.set_sense(self.problem.objective.sense.maximize)
        self.problem.write(os.path.join(root_path, 'debug.lp'))
        
    def solve(self):
        self.problem.solve()
        return self.problem.solution.get_values()


root_path = os.path.dirname(__file__)
data_folder = os.path.join(root_path, "data")
data_paths = [os.path.join(data_folder, name) for name in os.listdir(data_folder)]
dataset = Dataset(data_paths[1])
solver = MaxCliqueProblemSolver(dataset, root_path)
descision = solver.solve()
print(descision)