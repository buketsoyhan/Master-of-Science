import functions
from enumFunctions import Functions
from bb_bc import big_bang_big_crunch
import numpy as np
import pandas as pd

def run_bb_bc_trials():
    particle_sizes = [100,250,500,1000]
    num_iterations_list = [100,250,500,1000]
    beta_values = [0.1, 0.15, 0.2, 0.25, 0.3]

    results = []

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

    for func_enum, (lb, ub) in objective_functions.items():
        func_name = Functions(func_enum).name
        obj_func = functions.selectFunction(func_enum)

        for particle_size in particle_sizes:
            for num_iterations in num_iterations_list:
                for beta in beta_values:
                    best_fitness_values = []

                    for i in range(5):  
                        print(f"\nRunning with Particle Size = {particle_size}, Num Iterations = {num_iterations}, Beta = {beta}")
                        best_fitness_values.append(min(big_bang_big_crunch(particle_size, 30, num_iterations, beta, lb, ub, obj_func)))

                    best_fitness = min(best_fitness_values)
                    avg_fitness = np.mean(best_fitness_values)
                    std_dev_fitness = np.std(best_fitness_values)

                    results.append({
                        'Algorithm': "BB-BC",
                        'Function': func_name,
                        'Particle_Size': particle_size,
                        'Num_of_Iterations': num_iterations,
                        'Beta_Value': beta,
                        'Best_Fitness_1': best_fitness_values[0],
                        'Best_Fitness_2': best_fitness_values[1],
                        'Best_Fitness_3': best_fitness_values[2],
                        'Best_Fitness_4': best_fitness_values[3],
                        'Best_Fitness_5': best_fitness_values[4],
                        'Best_Fitness': best_fitness,
                        'Avg_Fitness': avg_fitness,
                        'Std_Dev_Fitness': std_dev_fitness,
                    })

    df_results = pd.DataFrame(results)

    df_results.to_excel("bb_bc_results_final.xlsx", index=False)

if __name__ == "__main__":
    run_bb_bc_trials()
