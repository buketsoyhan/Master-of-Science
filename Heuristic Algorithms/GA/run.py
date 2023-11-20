import functions
from enumFunctions import Functions
from GA import GA
import numpy as np
import pandas as pd

deger=(4*4*5*3*2*5)/100

def run_ga_combinations():
    pop_sizes = [100,250,500,1000]
    num_of_generations_values = [100,250,500,1000]
    mut_probabilities = [0.01, 0.02, 0.05, 0.1, 0.15]
    crossover_types = ['1-point', '2-point', 'uniform']
    selection_types = ['roulette_wheel', 'tournament_selection']

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
        sayac=0
        for pop_size in pop_sizes:
            for num_of_generations in num_of_generations_values:
                for mut_prob in mut_probabilities:
                    for crossover_type in crossover_types:
                        for selection_type in selection_types:
                            best_fitness_values = []
                            for _ in range(5):  
                                obj_func = functions.selectFunction(func_enum)

                                ga_instance = GA(obj_func, lb, ub, 30, pop_size, num_of_generations, mut_prob, crossover_type, selection_type)

                                result = ga_instance

                                best_fitness_values.append(result.best)
                                sayac=sayac+1
                                print(func_enum.name,((sayac/deger)))
                            avg_fitness = np.mean(best_fitness_values)
                            std_dev_fitness = np.std(best_fitness_values)
                            best_fitness = min(best_fitness_values)

                            results.append({
                                'Algorithm': "GA",
                                'Function': func_enum.name,
                                'Pop_Size': pop_size,
                                'Num_of_Generations': num_of_generations,
                                'Mut_Probability': mut_prob,
                                'Crossover_Type': crossover_type,
                                'Selection_Type': selection_type,
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
    df.to_excel('ga_results.xlsx', index=False)

if __name__ == "__main__":
    run_ga_combinations()
