import numpy as np

def fitness_function(x):
    return np.sum(x**2)

def initialize_particles(num_particles, dimension, lb, ub, beta):
    initial_particles = lb + np.random.rand(num_particles, dimension) * (ub - lb)

    modified_particles = initial_particles * (1 - beta) + beta * (lb + ub) / 2

    return modified_particles

def big_bang_big_crunch(num_particles, dimension, num_iterations, beta, lb, ub):
    particles = initialize_particles(num_particles, dimension, lb, ub, beta)
    
    best_fitness_values = []

    for iteration in range(num_iterations):
        fitness_values = np.apply_along_axis(fitness_function, 1, particles)
            
        best_solution_index = np.argmin(fitness_values)
        best_solution = particles[best_solution_index]
        best_fitness = fitness_values[best_solution_index]

        for i in range(num_particles):
            if i != best_solution_index:
                particles[i] = initialize_particles(1, dimension, lb, ub, beta)[0]
            
        best_fitness_values.append(best_fitness)

    return best_fitness_values
