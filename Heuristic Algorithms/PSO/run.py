import functions
from enumFunctions import Functions
from PSO import PSO
import pandas as pd
import numpy as np

def pso(obj_func, lb, ub, dim, pop_size, num_of_generations):
    print(f'Running PSO with func={obj_func.__name__}, pop_size={pop_size}, num_of_generations={num_of_generations}')
    
    best_fitness_values = []
    
    for i in range(5): 
        pso_instance = PSO(obj_func, lb, ub, dim, pop_size, num_of_generations)  
        best_solution = pso_instance.best
        best_fitness_values.append(best_solution)
        print(f"Iteration {i+1} - Best Fitness: {best_solution}")

    return best_fitness_values

def run_pso_trials():
    pop_sizes = [100,250,500,1000]
    num_of_generations_values = [100,250,500,1000]

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
                best_fitness_values = pso(obj_func, lb, ub, 30, pop_size, num_of_generations)

                best_fitness = min(best_fitness_values)
                avg_fitness = np.mean(best_fitness_values)
                std_dev_fitness = np.std(best_fitness_values)

                results.append({
                    'Algorithm': "PSO",
                    'Function': func_name,
                    'Pop_Size': pop_size,
                    'Num_of_Generations': num_of_generations,
                    'Best_Fitness_1': best_fitness_values[0],
                    'Best_Fitness_2': best_fitness_values[1],
                    'Best_Fitness_3': best_fitness_values[2],
                    'Best_Fitness_4': best_fitness_values[3],
                    'Best_Fitness_5': best_fitness_values[4],
                    'Best_Fitness': best_fitness,
                    'Avg_Fitness': avg_fitness,
                    'Std_Dev_Fitness': std_dev_fitness,
                })

    df = pd.DataFrame(results)
    df.to_excel('pso_results_final.xlsx', index=False)

if __name__ == "__main__":
    run_pso_trials()
