import functions
from enumFunctions import Functions
from GWO import GWO 
import pandas as pd
import numpy as np

def run_gwo_trials():
    pop_sizes = [100, 250, 500, 1000]
    num_of_generations_values = [100, 250, 500, 1000]
    a_values = [4, 3, 2]

    objective_functions = {
        Functions.ackley: (-32.768, 32.768),
        Functions.griewank: (-600, 600),
        Functions.schwefel: (-500, 500),
        Functions.rastrigin: (-5.12, 5.12),
        Functions.sphere: (-5.12, 5.12),
        Functions.perm: (-30, 30),
        Functions.zakharov: (-5, 10),
        Functions.rosenbrock: (-2048, 2048),
        Functions.dixonprice: (-10, 10),
    }

    results = []

    for func_enum, (lb, ub) in objective_functions.items():
        func_name = Functions(func_enum).name 
        obj_func = functions.selectFunction(func_enum)

        for pop_size in pop_sizes:
            for num_of_generations in num_of_generations_values:
                for a in a_values:
                    fitness_values = GWO(obj_func, lb, ub, 30, pop_size, num_of_generations, [a])

                    for solution_obj in fitness_values:
                        results.append({
                            'Algorithm': "GWO",
                            'Function': obj_func.__name__,
                            'Pop_Size': pop_size,
                            'Num_of_Generations': num_of_generations,
                            'A_Value': a,
                            'Best_Fitness': solution_obj.convergence[-1], 
                        })

    df = pd.DataFrame(results)
    df.to_excel('gwo_results_0.xlsx', index=False)

if __name__ == "__main__":
    run_gwo_trials()
