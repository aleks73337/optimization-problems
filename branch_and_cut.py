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

import sys
sys.setrecursionlimit(1500)

class BranchAndCut():

    @staticmethod
    def validate_integer_solution(solution):
        epsilon = 10e-6
        for el in solution:
            if (not np.isclose(el, 1, atol = epsilon)) & (not np.isclose(el, 0, atol = epsilon)):
                return False
        return True

    @staticmethod
    def check_solution(problem, dataset):
        descision = problem.solution.get_values()
        descision = enumerate(descision)
        clique = []
        epsilon = 0.00001
        for idx, val in descision:
            if np.isclose(val, 1, atol = epsilon):
                clique.append(idx)
        bad_variables = []
        for i in clique:
            for j in clique:
                if (i != j):
                    if (dataset.graph[i, j] != 1):
                        bad_variables.append([i, j])
        return bad_variables

    @staticmethod
    def separation(dataset : Dataset, descision : list):
        graph = dataset.graph
        descision = enumerate(descision)
        descision = sorted(descision, key = lambda x: x[1], reverse=True)
        start_idx = descision.pop(0)[0]
        result = [start_idx]
        check_if_independent = lambda x: sum([graph[x, i] for i in result]) == 0
        while(descision):
            idx, val = descision.pop(0)
            if check_if_independent(idx):
                result.append(idx)
        return result

    @staticmethod
    def get_branching_var(solution):
        res =  np.argmax([el if ((el != 1.0)) else -100 for el in solution])
        return res

    @staticmethod
    def solve(problem : cplex.Cplex, dataset : Dataset):
        global best_solution, best_score
        # Ищем решение с текущими ограничениями
        problem.solve()
        # Если решение не удовлетворяет ограничениям, не идем в эту ветку
        if (problem.solution.get_status() is not 1):
            print("Broken model")
            return None
        obj = problem.solution.get_objective_value()
        # Если решение хуже лучшего, не идем в эту ветку
        if (obj <= best_score):
            return None
        prev_ans = obj
        add_constr = lambda constraint: problem.linear_constraints.add(
                                            lin_expr = [cplex.SparsePair(constraint, val = [1.0] * len(constraint))],
                                            rhs = [1.0],
                                            names = ["c_" + str(np.random.randint(low = 10, high=1000000))],
                                            senses = ['L']
                                            )
        while True:
            constraint  = BranchAndCut.separation(dataset, problem.solution.get_values())
            if (len(constraint) <= 1):
                break
            add_constr(constraint)

            problem.solve()
            obj = problem.solution.get_objective_value()
            if (obj < best_score):
                return None
            elif np.isclose(prev_ans, obj, atol = 10e-6):
                break
            prev_ans = obj

        solution = problem.solution.get_values()
        if BranchAndCut.validate_integer_solution(solution):
            constraints = BranchAndCut.check_solution(problem, dataset)
            if len(constraints) == 0:
                if obj > best_score:
                    best_solution = solution
                    best_score = obj
                    print("New best result: ", best_score)
            else:
                for constr in constraints:
                    add_constr(constr)
                BranchAndCut.solve(problem, dataset)
            return
        
        branching_var = BranchAndCut.get_branching_var(solution)
        branching_var = [int(branching_var)]
        for val in [1.0, 0.0]:
            constr = problem.linear_constraints.add(
                                            lin_expr = [cplex.SparsePair(branching_var, val = [1.0])],
                                            rhs = [val],
                                            names = ["e_" + str(np.random.randint(low = 10, high=1000000))],
                                            senses = ['E'])
            BranchAndCut.solve(problem, dataset)
            problem.linear_constraints.delete(constr)

class BranchAndCutSolver():
    def __init__(self, dataset : np.array, root_path : str):
        self.base_problem = cplex.Cplex()
        self.base_problem.set_results_stream(None)
        self.dataset = dataset
        self.used_candidates = []
        # Переменные - это вершины. 0 - не берем вершину в макс клику, 1 - берем
        variables = ['y' + str(i) for i in range(dataset.number_of_vertices)]
        self.base_problem.variables.add(names = variables)

        # Добавляю ограничения вида 0 <= x <= 1
        for i, variable in enumerate(variables):
            self.base_problem.variables.set_lower_bounds(i, 0.0)
            self.base_problem.variables.set_upper_bounds(i, 1.0)

        # Objective - максимизируем к-во элементов в клике
        for variable in variables:
                self.base_problem.objective.set_linear([(variable, 1.0)])

        print("Start")
        lin_exprs = []
        # Ищем все вершины, которые не связаны ребром
        rows, cols = np.where(dataset.graph == 0)
        for i,j in zip(rows, cols):
            if (i != j):
                lin_exprs.append(cplex.SparsePair(ind = [int(i), int(j)], val = [1.0, 1.0]))

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
        return(BranchAndCut.solve(self.base_problem, self.dataset))

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
    for path in data_paths:
        graph_name = ''.join(path.split(os.sep)[-1].split('.')[0:-1])
        print(graph_name)
        res_path = os.path.join(descision_folder, graph_name + '.txt')
        dataset = Dataset(path)
        best_solution = 0
        best_score = 0
        solver = BranchAndCutSolver(dataset, root_path)
        start_time = time.time()
        try:
            doitReturnValue = func_timeout(1800, solver, args=())
        except Exception as e:
            print(e)
        total_time = time.time() - start_time
        print(total_time)

        def save_results(descision, calc_time, n_vertices, path):
            with open(path, 'w+') as f:
                res_string = "N vertices: {} \nCalculation time: {} seconds \nDescision: {}".format(n_vertices, calc_time, descision)
                f.write(res_string)

        try:
            save_results(best_solution, total_time, best_score, res_path)
        except Exception as e:
            print(e)