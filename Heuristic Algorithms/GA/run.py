import pandas as pd
import functions
from enumFunctions import Functions
import GA

def GeneticAlgorithm(pop_size, iteration_number):
    functionIndex = Functions.griewank
    _lb = -600
    _ub = 600
    dim = 30
    maxiter = iteration_number
    obj_func = functions.selectFunction(Functions.griewank)
    sol = GA.GA(obj_func, _lb, _ub, dim, pop_size, maxiter)
    return sol

def main():
    pop_size_values = [100, 250, 500, 1000]
    iteration_number_values = [100, 250, 500, 1000]

    results = []  # Sonuçları saklamak için bir liste

    for pop_size in pop_size_values:
        for iteration_number in iteration_number_values:
            sol = GeneticAlgorithm(pop_size, iteration_number)
            result = {
                'pop_size': pop_size,
                'iteration_number': iteration_number,
                'best_fitness': sol.best,
                'execution_time': sol.executionTime
            }
            results.append(result)

    # Sonuçları bir DataFrame'e dönüştürün ve bir Excel dosyasına kaydedin
    result_df = pd.DataFrame(results)
    result_df.to_excel('GA_Results.xlsx', index=False)

if __name__ == "__main__":
    main()