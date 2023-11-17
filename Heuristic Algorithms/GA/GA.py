"""
Created on Sat Feb  24 20:18:05 2019

@author: Raneem
"""
import numpy as np
import random
import time
import sys

from solution import solution

def crossoverPopulaton(population, scores, popSize, crossoverProbability, keep):
    newPopulation = np.empty_like(population)
    newPopulation[0:keep] = population[0:keep]

    for i in range(keep, popSize, 2):
        if i + 1 < popSize:  # Eğer i+1 popSize'dan küçükse devam et
            parent1, parent2 = pairSelection(population, scores, popSize)
            crossoverLength = min(len(parent1), len(parent2))
            parentsCrossoverProbability = random.uniform(0.0, 1.0)

            if parentsCrossoverProbability < crossoverProbability:
                offspring1, offspring2 = crossover(crossoverLength, parent1, parent2)
            else:
                offspring1 = parent1.copy()
                offspring2 = parent2.copy()

            newPopulation[i] = np.copy(offspring1)
            newPopulation[i + 1] = np.copy(offspring2)

    return newPopulation

def mutatePopulaton(population, popSize, mutationProbability, keep, lb, ub):
    for i in range(keep, popSize):
        offspringMutationProbability = random.uniform(0.0, 1.0)
        if offspringMutationProbability < mutationProbability:
            mutation(population[i], len(population[i]), lb, ub)

def elitism(population, scores, bestIndividual, bestScore):
    worstFitnessId = selectWorstIndividual(scores)

    if scores[worstFitnessId] > bestScore:
        population[worstFitnessId] = np.copy(bestIndividual)
        scores[worstFitnessId] = np.copy(bestScore)

def selectWorstIndividual(scores):
    maxFitnessId = np.where(scores == np.max(scores))
    maxFitnessId = maxFitnessId[0][0]
    return maxFitnessId

def pairSelection(population, scores, popSize):
    parent1Id = rouletteWheelSelectionId(scores, popSize)
    parent1 = population[parent1Id].copy()

    parent2Id = rouletteWheelSelectionId(scores, popSize)
    parent2 = population[parent2Id].copy()

    return parent1, parent2

def rouletteWheelSelectionId(scores, popSize):
    reverse = max(scores) + min(scores)
    reverseScores = reverse - scores.copy()
    sumScores = sum(reverseScores)
    pick = random.uniform(0, sumScores)
    current = 0

    for individualId in range(popSize):
        current += reverseScores[individualId]
        if current > pick:
            return individualId

def crossover(individualLength, parent1, parent2):
    if individualLength <= 1:
        return parent1.copy(), parent2.copy()

    crossover_point = random.randint(1, individualLength - 1)
    crossover_point2 = random.randint(1, individualLength - 1)

    while crossover_point == crossover_point2:
        crossover_point2 = random.randint(1, individualLength - 1)

    if crossover_point > crossover_point2:
        temp = crossover_point2
        crossover_point2 = crossover_point
        crossover_point = temp

    offspring1 = np.concatenate([parent1[0:crossover_point], parent2[crossover_point:crossover_point2], parent1[crossover_point2:]])
    offspring2 = np.concatenate([parent2[0:crossover_point], parent1[crossover_point:crossover_point2], parent2[crossover_point2:]])

    return offspring1, offspring2

def mutation(offspring, individualLength, lb, ub):
    mutationIndex = random.randint(0, individualLength - 1)
    mutationValue = random.uniform(lb[mutationIndex], ub[mutationIndex])
    offspring[mutationIndex] = mutationValue

def clearDups(Population, lb, ub):
    newPopulation = np.unique(Population, axis=0)
    oldLen = len(Population)
    newLen = len(newPopulation)

    if newLen < oldLen:
        nDuplicates = oldLen - newLen
        newPopulation = np.append(newPopulation, np.random.uniform(0, 1, (nDuplicates, len(Population[0]))) * (np.array(ub) - np.array(lb)) + np.array(lb), axis=0)

    return newPopulation

def calculateCost(objf, population, popSize, lb, ub):
    scores = np.full(popSize, np.inf)

    for i in range(0, popSize):
        population[i] = np.clip(population[i], lb, ub)
        scores[i] = objf(population[i, :])

    return scores

def sortPopulation(population, scores):
    sortedIndices = scores.argsort()
    population = population[sortedIndices]
    scores = scores[sortedIndices]

    return population, scores

def GA(objf, lb, ub, dim, popSize, iters):
    cp = 1
    mp = 0.1
    keep = 2

    s = solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    bestIndividual = np.zeros(dim)
    scores = np.random.uniform(0.0, 1.0, popSize)
    bestScore = float("inf")

    ga = np.zeros((popSize, dim))

    for i in range(dim):
        ga[:, i] = np.random.uniform(0, 1, popSize) * (ub[i] - lb[i]) + lb[i]
    convergence_curve = np.zeros(iters)

    print('GA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(iters):
        ga = crossoverPopulaton(ga, scores, popSize, cp, keep)
        mutatePopulaton(ga, popSize, mp, keep, lb, ub)
        ga = clearDups(ga, lb, ub)
        scores = calculateCost(objf, ga, popSize, lb, ub)
        bestScore = min(scores)
        ga, scores = sortPopulation(ga, scores)
        convergence_curve[l] = bestScore

        if l % 1 == 0:
            print(["At iteration " + str(l + 1) + " the best fitness is " + str(bestScore)])
            s.result.append([l + 1, bestScore])
        print(l)

    timerEnd = time.time()
    s.bestIndividual = bestIndividual
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.best = bestScore
    return s



def read(objf, lb, ub, dim, popSize, iters, ga):

    """
    This is the main method which implements GA

    Parameters
    ----------
    objf : function
        The objective function selected
    lb: list
        lower bound limit list
    ub: list
        Upper bound limit list
    dim: int
        The dimension of the indivisual
    popSize: int
        Number of chrmosomes in a population
    iters: int
        Number of iterations / generations of GA

    Returns
    -------
    obj
        s: The solution obtained from running the algorithm
    """

    cp = 1  # crossover Probability
    mp = 0.01  # Mutation Probability
    keep = 2
    # elitism parameter: how many of the best individuals to keep from one generation to the next

    s = solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    bestIndividual = np.zeros(dim)
    scores = np.random.uniform(0.0, 1.0, popSize)
    bestScore = float("inf")

    convergence_curve = np.zeros(iters)

    print('GA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(iters):

        # crossover
        ga = crossoverPopulaton(ga, scores, popSize, cp, keep)

        # mutation
        mutatePopulaton(ga, popSize, mp, keep, lb, ub)

        ga = clearDups(ga, lb, ub)

        scores = calculateCost(objf, ga, popSize, lb, ub)

        bestScore = min(scores)

        # Sort from best to worst
        ga, scores = sortPopulation(ga, scores)

        convergence_curve[l] = bestScore

        if l % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(l + 1)
                    + " the best fitness is "
                    + str(bestScore)
                ]
            )
            s.result.append([l+1,bestScore])
        print(l)
    timerEnd = time.time()
    s.bestIndividual = bestIndividual
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "GA"
    s.objfname = objf.__name__

    return s