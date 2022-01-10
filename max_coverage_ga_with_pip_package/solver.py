import os
import io
import json

import numpy as np
import pandas as pd
import geneticalgorithm.geneticalgorithm as ga

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
FILES_DIR = os.path.join(ROOT_DIR, "files")

# Parse settings
file_path = os.path.join(FILES_DIR, "settings.json")
with open(file_path) as fp:
    inp = json.load(fp)

input_settings = inp["settings"]
max_coverage_distance = input_settings["max_coverage_distance"]
K = input_settings["max_facilities"]
N = input_settings["n_locations_demand"]
M = input_settings["n_locations_possible_facility"]
maximum_generations = input_settings["ga_settings"]["maximum_generations"]
mutation_probability = input_settings["ga_settings"]["mutation_probability"]
population_size = input_settings["ga_settings"]["population_size"]
crossover_probability = input_settings["ga_settings"]["crossover_probability"]
parents_portion = input_settings["ga_settings"]["parents_portion"]
elite_ratio = input_settings["ga_settings"]["elite_ratio"]
maximum_generations_without_improvement = input_settings["ga_settings"]["maximum_generations_without_improvement"]
crossover_type = input_settings["ga_settings"]["crossover_type"]

# Parse distances
file_path = os.path.join(FILES_DIR, "Distance Matrix.xlsx")
distances = pd.read_excel(file_path, "distances", header=None).values
coverage_matrix = np.where(distances <= max_coverage_distance, 1, 0)

# Parse demand
file_path = os.path.join(FILES_DIR, "Project Data.xlsx")
demand = pd.read_excel(file_path, "demand", header=None).values

# Parse capacity
file_path = os.path.join(FILES_DIR, "Project Data.xlsx")
capacity = pd.read_excel(file_path, "capacity", header=None).values

# Parse info
file_path = os.path.join(FILES_DIR, "Project Data.xlsx")
info = pd.read_excel(file_path, "info", header=None).values

np.random.seed(7)

def f(X):
    x = np.array(X[0:28])
    y = np.array(X[28:56])
    z = np.array(X[56:840])
    obj_value = np.sum(y * demand)

    coverage_set = coverage_matrix.flatten()

    # Demand coverage constraint penalty
    demand_coverage_penalty = 0.0
    pen_coverage = []
    for idx, yi in enumerate(y):
        if yi > np.sum(z):
            # TODO sum over its own set the whole set of facilities
            pen = 500 + 1000 * (yi - np.sum(z * coverage_set))
            pen_coverage.append(pen)
    demand_coverage_penalty = sum(pen_coverage)

    # Max number of facilities constraint penalty
    max_facility_penalty = 0.0
    if np.sum(x) > K:
        max_facility_penalty = 500 + 1000 * (np.sum(x) - K)

    # Capacity constraint penalty
    capacity_penalty = 0.0
    pen_capacity = []
    for idx, xi in enumerate(x):
        if  np.sum(demand * coverage_set * z) > capacity[idx] * xi:
            pen = capacity[idx] * xi - (np.sum(demand * coverage_set * z))
            pen_capacity.append(pen)
    capacity_penalty = sum(pen_capacity)

    # Assignment constraint penalty
    assignment_penalty = 0.0
    z = z.reshape((N, M))
    for j in range(M):
        zij = z[:][j]
        for i in range(N):
            if zij[i] > x[j]:
                assignment_penalty = assignment_penalty + 1000

    # Self assignment constraint penalty
    self_assignment_penalty = 0.0
    for i in range(N):
        if x[i] > y[i]:
            self_assignment_penalty = self_assignment_penalty + 1000

    return -obj_value * 1000 + demand_coverage_penalty + 5000 * max_facility_penalty + capacity_penalty + assignment_penalty + self_assignment_penalty

def report(output_dict):
    X_best = output_dict["variable"]
    f_best = output_dict["function"]
    x_optimal = pd.DataFrame((X_best[0:28]))
    y_optimal = pd.DataFrame((X_best[28:56]))
    z_optimal = pd.DataFrame(X_best[56:840].reshape([N, M]))

    print("x_optimal:", x_optimal, "\n", "y_optimal:", y_optimal, "\n", "z_optimal:", z_optimal, "\n")
    output_file = "results.xlsx"
    output_file_path = os.path.join(FILES_DIR, output_file)

    with pd.ExcelWriter(output_file_path) as writer:
        x_optimal.to_excel(writer, sheet_name='x')

    with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a') as writer:  
        y_optimal.to_excel(writer, sheet_name='y')
        z_optimal.to_excel(writer, sheet_name='z')

    facility_indexes = np.where(x_optimal == 1)
    locations_open = info[facility_indexes]

    demand_covered_indexes = np.where(y_optimal == 1)
    demand_locations_covered = info[demand_covered_indexes]

    demand_uncovered_indexes = np.where(y_optimal == 0)
    demand_locations_uncovered = info[demand_uncovered_indexes]

    summary_file = "summary.txt"
    summary_file_path = os.path.join(FILES_DIR, summary_file)
    with io.open(summary_file_path, "w", encoding="utf-8") as fp:
        fp.write("Genetic Algorithm Solver Summary\n")
        fp.write("x:\n")
        fp.write(x_optimal.to_string())
        fp.write("\n\n")
        fp.write("y:\n")
        fp.write(y_optimal.to_string())
        fp.write("\n\n")
        fp.write("z:\n")
        fp.write(z_optimal.to_string())
        fp.write("\n\n")
        fp.write("Locations Selected:\n")
        fp.write("-------------------\n")
        for loc in locations_open:
            fp.write(loc)
            fp.write("\n")

        fp.write("\n")
        fp.write("Demand Locations Covered:\n")
        fp.write("-------------------\n")
        for loc in demand_locations_covered:
            fp.write(loc)
            fp.write("\n")

        fp.write("\n")
        fp.write("Demand Locations Uncovered:\n")
        fp.write("-------------------\n")
        for loc in demand_locations_uncovered:
            fp.write(loc)
            fp.write("\n")


algorithm_param = {
    'max_num_iteration': maximum_generations,
    'population_size': population_size,
    'mutation_probability': mutation_probability,
    'elit_ratio': elite_ratio,
    'crossover_probability': crossover_probability,
    'parents_portion': parents_portion,
    'crossover_type':crossover_type,
    'max_iteration_without_improv':maximum_generations_without_improvement
}

model=ga(function=f, dimension=840, variable_type='bool', function_timeout=1000, algorithm_parameters=algorithm_param)

model.run()
output_dict = model.output_dict
report(output_dict)
