import os
import numpy as np
import cplex
import networkx
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import time
from func_timeout import func_timeout, FunctionTimedOut

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
            if (line[0] == 'p'):
                self.number_of_vertices = int(line.split(' ')[2])
                print(self.number_of_vertices)
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
        epsilon = 0.00001
        for el in solution:
            if ((np.abs(el - 1.0) > epsilon) & np.abs(el > epsilon)):
                return False
        return True

    @staticmethod
    def solve(problem : cplex.Cplex, dataset : Dataset, used_candidates : list):
        global best_solution, best_score
        problem.solve()
        descision = problem.solution.get_values()
        obj = problem.solution.get_objective_value()
        if (obj <= best_score):
            return None
        if (problem.solution.get_status() is not 1):
            return None
        new_candidates_1 = BranchAndBound.add_constraint(problem, dataset, used_candidates, 1.0)
        if (new_candidates_1 is not None):
            b_1 = BranchAndBound.solve(problem, dataset, copy.deepcopy(new_candidates_1))
            if (b_1 is not None):
                problem.solve()
                b_1_res = problem.solution.get_objective_value()
                if (not BranchAndBound.validate_solution(b_1)):
                    b_1 = descision
                    b_1_res = obj
                else:
                    if (best_score < b_1_res):
                        best_score = b_1_res
                        best_solution = b_1
                        print(best_score)
            else:
                b_1_res = obj
                b_1 = descision
            problem.linear_constraints.delete(problem.linear_constraints.get_num() - 1)
        else:
            b_1 = descision
            b_1_res = obj

        new_candidates_0 = BranchAndBound.add_constraint(problem, dataset, used_candidates, 0.0)
        if (new_candidates_0 is not None):
            b_0 = BranchAndBound.solve(problem, dataset, copy.deepcopy(new_candidates_0))
            if (b_0 is not None):
                problem.solve()
                b_0_res = problem.solution.get_objective_value()
                if (not BranchAndBound.validate_solution(b_0)):
                    b_0 = descision
                    b_0_res = obj
                else:
                    if (best_score < b_0_res):
                        best_score = b_0_res
                        best_solution = b_0
                        print(best_score)
            else:
                b_0 = descision
                b_0_res = obj
            problem.linear_constraints.delete(problem.linear_constraints.get_num() - 1)
        else:
            b_0 = descision
            b_0_res = obj

        if (b_0_res < b_1_res):
            return b_1
        elif (b_0_res > b_1_res):
            return b_0
        else:
            if (BranchAndBound.validate_solution(descision)):
                return descision
            else:
                return None

    @staticmethod
    def add_constraint(problem : cplex.Cplex, dataset : Dataset, used_candidates : list, constr_val):
        for i in range(dataset.number_of_vertices):
            if (i not in used_candidates):
                used_candidates.append(i)
                indx = problem.linear_constraints.add(
                    lin_expr = [cplex.SparsePair([i], val = [1.0])],
                    rhs = [constr_val],
                    names = ["c_" + str(np.random.randint(low = 10)) + "_" + str(np.random.randint(low = 10))],
                    senses = ["G"] if (constr_val == 1.0) else ["L"] 
                )
                # print("Added {}".format(indx))
                used_candidates.append(i)
                return used_candidates
        return None

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

        print("Start")
        lin_exprs = []
        rows, cols = np.where(dataset.graph == 0)
        for i,j in zip(rows, cols):
            if (i != j):
                lin_exprs.append(cplex.SparsePair(ind = [int(i), int(j)], val = [1.0, 1.0]))
                for k in range(dataset.number_of_vertices):
                    if ((i != k) & (j != k)):
                        if (dataset[i, j] + dataset[i, k] + dataset[j, k] == 0):
                            lin_exprs.append(cplex.SparsePair(ind = [int(i), int(j), int(k)], val = [1.0, 1.0, 1.0]))

        print("Stop")

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
    
    # def __name__(self):
    #     return "BranchAndBoundSolver"



root_path = os.path.dirname(__file__)
data_folder = os.path.join(root_path, "data")
data_paths = [os.path.join(data_folder, name) for name in os.listdir(data_folder)]

descision_folder = os.path.join(root_path, "results")
if not os.path.isdir(descision_folder):
    os.makedirs(descision_folder)

global best_solution, best_score
for path in data_paths[5:]:
    graph_name = path.split(os.sep)[-1]
    print(graph_name)
    res_path = os.path.join(descision_folder, graph_name + '.txt')
    dataset = Dataset(path)
    best_solution = 0
    best_score = 0
    solver = BranchAndBoundSolver(dataset, root_path)
    start_time = time.time()
    try:
        doitReturnValue = func_timeout(1800, solver, args=())
    except Exception as e:
        print(e)
    total_time = time.time() - start_time
    print(total_time)

    def save_results(descision, calc_time, n_vertices, path):
        with open(path, 'w+') as f:
            res_string = "N vertices: {} \n Calculation time: {} seconds \n Descision: {}".format(n_vertices, calc_time, descision)
            f.write(res_string)

    def check_if_clique(descision, graph):
        for i in descision:
            for j in descision:
                if (i != j):
                    if (graph[int(i), int(j)] != 1):
                        return False
        return True

    try:
        print(check_if_clique(best_solution, dataset.graph))
        save_results(best_solution, total_time, best_score, res_path)
    except Exception as e:
        print(e)