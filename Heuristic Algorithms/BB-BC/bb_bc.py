# bb_bc.py
import numpy as np

def fitness_function(x, obj_func):
    return obj_func(x)

def initialize_particles(num_particles, dimension, lb, ub, beta):
    # Başlangıç noktalarını oluştur
    initial_particles = lb + np.random.rand(num_particles, dimension) * (ub - lb)

    # Başlangıç noktalarını Big Bang-Big Crunch algoritmasına göre düzenle
    modified_particles = initial_particles * (1 - beta) + beta * (lb + ub) / 2

    return modified_particles

def big_bang_big_crunch(num_particles, dimension, num_iterations, beta, lb, ub, obj_func):
    particles = initialize_particles(num_particles, dimension, lb, ub, beta)
    
    best_fitness_values = []

    for iteration in range(num_iterations):
        fitness_values = np.apply_along_axis(fitness_function, 1, particles, obj_func)
            
        best_solution_index = np.argmin(fitness_values)
        best_solution = particles[best_solution_index]
        best_fitness = fitness_values[best_solution_index]

        for i in range(num_particles):
            if i != best_solution_index:
                # Yeni konumlar lb ve ub değerleri arasında rastgele belirlenir
                particles[i] = initialize_particles(1, dimension, lb, ub, beta)[0]
            
        best_fitness_values.append(best_fitness)

    return best_fitness_values
