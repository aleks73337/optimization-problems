import os
import numpy as np


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
                self.graph[vertex_from, vertex_to] = 1
                self.graph[vertex_to, vertex_from] = 1
        self.graph = self.graph.tolist()


class MaxCliqueProblemSolver():
    def __init__(self, dataset : Dataset):
        self.problem = cplex.Cplex()
        # self.coefficients = [item for sublist in dataset.graph for item in sublist]
        self.variables = [[str(i) + '_' + str(j) for j in range(dataset.number_of_vertices)] for i in range(dataset.number_of_vertices)]
        # self.variables = [item for sublist in self.variables for item in sublist]
        for variables in self.variables:
            self.problem.variables.add(names= variables)
        print(self.problem.variables)


root_path = os.path.dirname(__file__)
data_folder = os.path.join(root_path, "data")
data_paths = [os.path.join(data_folder, name) for name in os.listdir(data_folder)]
dataset = Dataset(data_paths[0])
MaxCliqueProblemSolver(dataset)