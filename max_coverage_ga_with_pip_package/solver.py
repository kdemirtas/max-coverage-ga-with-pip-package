import os
import json

import numpy as np
import pandas as pd
import geneticalgorithm.geneticalgorithm as ga

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
FILES_DIR = os.path.join(ROOT_DIR, "files")

# Parse distances
file_path = os.path.join(FILES_DIR, "Distance Matrix.xlsx")
distances = pd.read_excel(file_path, "distances", header=None).values

# Parse demand
file_path = os.path.join(FILES_DIR, "Project Data.xlsx")
demand = pd.read_excel(file_path, "demand", header=None).values

# Parse capacity
file_path = os.path.join(FILES_DIR, "Project Data.xlsx")
capacity = pd.read_excel(file_path, "capacity", header=None).values

# Parse settings
file_path = os.path.join(FILES_DIR, "settings.json")
with open(file_path) as fp:
    inp = json.load(fp)

input_settings = inp["settings"]


def f(X):
    yi = X[28:55]
    obj_value = np.sum(yi * demand)

    # Coverage constraint penalty

    return -obj_value

algorithm_param = {
    'max_num_iteration': 1000,
    'population_size':100,
    'mutation_probability':0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type':'uniform',
    'max_iteration_without_improv':100
}

model=ga(function=f, dimension=840, variable_type='bool', function_timeout=1000, algorithm_parameters=algorithm_param)
output_dict = model.run()
print("Done")