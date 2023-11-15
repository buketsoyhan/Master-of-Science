import functions
from enumFunctions import Functions
from PSO import PSO
import pandas as pd
import numpy as np

def pso(obj_func, lb, ub, dim, pop_size, num_of_generations, num_runs=10):
    print(f'Running PSO with func={obj_func.__name__}, pop_size={pop_size}, num_of_generations={num_of_generations}')
    fitness_values = []
    for _ in range(num_runs):
        pso_instance = PSO(obj_func, lb, ub, dim, pop_size, num_of_generations)
        best_solution = pso_instance.best
        fitness_values.append(best_solution)
    return fitness_values

def run_pso_trials():
    pop_sizes = [100, 250, 500, 1000]
    num_of_generations_values = [100, 250, 500, 1000]

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
                fitness_values = pso(obj_func, lb, ub, 30, pop_size, num_of_generations)
                avg_fitness = np.mean(fitness_values)
                std_dev = np.std(fitness_values)

                results.append({
                    'Algorithm': "PSO",
                    'Function': func_name,
                    'Pop_Size': pop_size,
                    'Num_of_Generations': num_of_generations,
                    'Best_Fitness': fitness_values[0],  
                    'Avg_Fitness': avg_fitness,
                    'Std_Dev': std_dev
                })

    df = pd.DataFrame(results)
    df.to_excel('pso_results_2.xlsx', index=False)

if __name__ == "__main__":
    run_pso_trials()

