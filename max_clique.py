import os
import numpy as np
import cplex
import networkx
import matplotlib.pyplot as plt
import copy

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
        # self.graph = self.graph.tolist()

    def __getitem__(self, a):
        return self.graph[a]


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

class BranchAndBound():
    i = 0

    @staticmethod
    def validate_solution(solution):
        for i in solution:
            if ((i != 1.0) & (i != 0.0)):
                return False
        return True

    @staticmethod
    def solve(problem : cplex.Cplex, dataset : Dataset, used_candidates : list):
        success = BranchAndBound.add_constraint(problem, dataset, used_candidates)
        if not success:
            return None
        problem.solve()
        if (not problem.solution.get_status()):
            return None
        descision = problem.solution.get_values()
        if (BranchAndBound.validate_solution(problem.solution.get_values())):
            return descision
        obj_value = problem.solution.get_objective_value()
        print(obj_value)
        BranchAndBound.i += 1
        if (BranchAndBound.i < 50):
            b = BranchAndBound.solve(problem, dataset, used_candidates)
        return b
    
    @staticmethod
    def find_candidate(dataset : Dataset, used_candidates : list):
        # for i  in range(dataset.number_of_vertices):
        #     for j in range(i, dataset.number_of_vertices):
        #         if (i != j):
        #             if (dataset[i, j] == 0):
        #                 for k in range(dataset.number_of_vertices):
        #                     if ((i != j) & (j != k) & (i != k)):
        #                         if ((dataset[i, k] == 0) & (dataset[i, j] == 0)):
        #                             if (([i, j, k] not in used_candidates)):
        #                                 return [i, j, k]
        row_idxs, col_idxs = np.where(dataset.graph == 0)
        for i, j in zip(row_idxs, col_idxs):
            if (i != j):
                for k in range(dataset.number_of_vertices):
                    if ((j != k) & (i != k)):
                        if ((dataset[i, k] == 0) & (dataset[i, j] == 0)):
                            if (([i, j, k] not in used_candidates)):
                                return [i, j, k]
        return None

    @staticmethod
    def add_constraint(problem : cplex.Cplex, dataset : Dataset, used_candidates : list):
        candidates = BranchAndBound.find_candidate(dataset, used_candidates)
        print(candidates)
        if (candidates):
            used_candidates.append(candidates)
            candidates_len = len(candidates)
            problem.linear_constraints.add(
                lin_expr = [cplex.SparsePair(candidates, val = [1.0] * candidates_len)],
                rhs = [1.0],
                names = ["c_" + str(np.random.randint(low = 10)) + "_" + str(np.random.randint(low = 10))],
                senses = ["L"]
            )
            return True
        else:
            return False

class BranchAndBoundSolver():
    def __init__(self, dataset : np.array, root_path : str):
        self.base_problem = cplex.Cplex()
        self.base_problem.set_results_stream(None)
        self.dataset = dataset
        self.used_candidates = []
        variables = ['y' + str(i) for i in range(dataset.number_of_vertices)]
        self.base_problem.variables.add(names = variables)

        for i, variable in enumerate(variables):
            self.base_problem.variables.set_lower_bounds(i, 0.0)
            self.base_problem.variables.set_upper_bounds(i, 1.0)

        for variable in variables:
                self.base_problem.objective.set_linear([(variable, 1.0)])

        lin_exprs = []
        for i in range(dataset.number_of_vertices):
            for j in range(dataset.number_of_vertices):
                if ((dataset.graph[i, j] == 0) & (i != j)):
                    lin_exprs.append(cplex.SparsePair(ind = [i, j], val = [1.0, 1.0]))

        lin_exprs_len = len(lin_exprs)
        self.base_problem.linear_constraints.add(
            lin_expr = lin_exprs,
            rhs = [1.0] * lin_exprs_len,
            names = ["y_" + str(i) for i in range(lin_exprs_len)],
            senses = ["L"] * lin_exprs_len
        )

        self.base_problem.objective.set_sense(self.base_problem.objective.sense.maximize)
    
    def __call__(self):
        return(BranchAndBound.solve(self.base_problem, self.dataset, self.used_candidates))


root_path = os.path.dirname(__file__)
data_folder = os.path.join(root_path, "data")
data_paths = [os.path.join(data_folder, name) for name in os.listdir(data_folder)]
dataset = Dataset(data_paths[1])
solver = BranchAndBoundSolver(dataset, root_path)
print(solver())