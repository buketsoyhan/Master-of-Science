import numpy
import random
import time
import sys
from enumFunctions import Functions
from solution import solution
import functions

tournament_size = 2

def crossoverPopulaton(population, scores, popSize, crossoverProbability, keep, lb, ub, selection_type, crossover_type, mut_prob):
    newPopulation = numpy.empty_like(population)
    newPopulation[0:keep] = population[0:keep]

    for i in range(keep, popSize, 2):
        #(population, scores, popSize, selection_type, tournament_size):
        parent1, parent2 = pairSelection(population, scores, popSize, selection_type, 2)
        crossoverLength = min(len(parent1), len(parent2))
        parentsCrossoverProbability = random.uniform(0.0, 1.0)
        
        if parentsCrossoverProbability < crossoverProbability:
            offspring1, offspring2 = crossover(crossoverLength, parent1, parent2, crossover_type, mut_prob)
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()

        newPopulation[i] = numpy.copy(offspring1)
        newPopulation[i + 1] = numpy.copy(offspring2)

    return newPopulation

def mutatePopulaton(population, popSize, mut_prob, keep, lb, ub):
    for i in range(keep, popSize):
        offspringMutationProbability = random.uniform(0.0, 1.0)
        
        if offspringMutationProbability < mut_prob:
            mutation(population[i], len(population[i]), lb, ub)

def elitism(population, scores, bestIndividual, bestScore):
    worstFitnessId = selectWorstIndividual(scores)

    if scores[worstFitnessId] > bestScore:
        population[worstFitnessId] = numpy.copy(bestIndividual)
        scores[worstFitnessId] = numpy.copy(bestScore)

def selectWorstIndividual(scores):
    """
    It is used to get the worst individual in a population based n the fitness value

    Parameters
    ----------
    scores : list
        The list of fitness values for each individual

    Returns
    -------
    int
        maxFitnessId: The individual id of the worst fitness value
    """

    maxFitnessId = numpy.where(scores == numpy.max(scores))
    maxFitnessId = maxFitnessId[0][0]
    return maxFitnessId


def pairSelection(population, scores, popSize, selection_type, tournament_size):
    if selection_type == 'roulette_wheel':
        return pairSelectionRoulette(population, scores, popSize)
    elif selection_type == 'tournament_selection':
        return pairSelectionTournament(population, scores, tournament_size)

def pairSelectionRoulette(population, scores, popSize):
    parent1Id = rouletteWheelSelectionId(scores, popSize)
    parent1 = population[parent1Id].copy()

    parent2Id = rouletteWheelSelectionId(scores, popSize)
    parent2 = population[parent2Id].copy()

    return parent1, parent2

def rouletteWheelSelectionId(scores, popSize):
    """
    A roulette Wheel Selection mechanism for selecting an individual

    Parameters
    ----------
    scores : list
        The list of fitness values for each individual
    popSize: int
        Number of chrmosome in a population

    Returns
    -------
    id
        individualId: The id of the individual selected
    """

    ##reverse score because minimum value should have more chance of selection
    reverse = max(scores) + min(scores)
    reverseScores = reverse - scores.copy()
    sumScores = sum(reverseScores)
    pick = random.uniform(0, sumScores)
    current = 0
    for individualId in range(popSize):
        current += reverseScores[individualId]
        if current > pick:
            return individualId


def pairSelectionTournament(population, scores, tournament_size):
    parent1Id = tournament_selection(population, scores, tournament_size)
    parent1 = population[parent1Id].copy()

    parent2Id = tournament_selection(population, scores, tournament_size)
    parent2 = population[parent2Id].copy()

    return parent1, parent2

def tournament_selection(population, scores, tournament_size):
    tournament_candidates = random.sample(range(len(population)), tournament_size)
    selected_index = max(tournament_candidates, key=lambda x: scores[x])
    return selected_index

def crossover(individualLength, parent1, parent2, crossover_type, mut_prob):
    if crossover_type == '1-point':
        return crossoverOnePoint(individualLength, parent1, parent2)
    elif crossover_type == '2-point':
        return crossoverTwoPoint(individualLength, parent1, parent2)
    elif crossover_type == 'uniform':
        return crossoverUniform(individualLength, parent1, parent2, mut_prob)

def crossoverOnePoint(individualLength, parent1, parent2):
    crossover_point = random.randint(0, individualLength - 1)
    offspring1 = numpy.concatenate([parent1[0:crossover_point], parent2[crossover_point:]])
    offspring2 = numpy.concatenate([parent2[0:crossover_point], parent1[crossover_point:]])
    return offspring1, offspring2

def crossoverTwoPoint(individualLength, parent1, parent2):
    crossover_point = random.randint(1, individualLength - 1)
    crossover_point2 = random.randint(1, individualLength - 1)

    while crossover_point == crossover_point2:
        crossover_point2 = random.randint(0, individualLength - 1)

    if crossover_point > crossover_point2:
        temp = crossover_point2
        crossover_point2 = crossover_point
        crossover_point = temp

    offspring1 = numpy.concatenate([parent1[0:crossover_point], parent2[crossover_point:crossover_point2], parent1[crossover_point2:]])
    offspring2 = numpy.concatenate([parent2[0:crossover_point], parent1[crossover_point:crossover_point2], parent2[crossover_point2:]])
    return offspring1, offspring2

def crossoverUniform(individualLength, parent1, parent2, mut_prob):
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    for i in range(individualLength):
        if random.random() < mut_prob:
            offspring1[i], offspring2[i] = offspring2[i], offspring1[i]

    return offspring1, offspring2

def mutation(offspring, individualLength, lb, ub):
    mutationIndex = random.randint(0, individualLength - 1)
    mutationValue = random.uniform(lb[mutationIndex], ub[mutationIndex])
    offspring[mutationIndex] = mutationValue

def clearDups(Population, lb, ub):
    newPopulation = numpy.unique(Population, axis=0)
    oldLen = len(Population)
    newLen = len(newPopulation)
    
    if newLen < oldLen:
        nDuplicates = oldLen - newLen
        newPopulation = numpy.append(
            newPopulation,
            numpy.random.uniform(0, 1, (nDuplicates, len(Population[0])))
            * (numpy.array(ub) - numpy.array(lb))
            + numpy.array(lb),
            axis=0,
        )

    return newPopulation

def calculateCost(objf, population, popSize, lb, ub):
    scores = numpy.full(popSize, numpy.inf)

    for i in range(0, popSize):
        population[i] = numpy.clip(population[i], lb, ub)
        scores[i] = objf(population[i, :])

    return scores

def sortPopulation(population, scores):
    sortedIndices = scores.argsort()
    population = population[sortedIndices]
    scores = scores[sortedIndices]
    return population, scores

def GA(objf, lb, ub, dim, popSize, num_of_generations, mut_prob, crossover_type, selection_type):
    cp = 1  # crossover Probability
    mp = mut_prob
    keep = 2

    s = solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    bestIndividual = numpy.zeros(dim)
    scores = numpy.random.uniform(0.0, 1.0, popSize)
    bestScore = float("inf")

    ga = numpy.zeros((popSize, dim))
    for i in range(dim):
        ga[:, i] = numpy.random.uniform(0, 1, popSize) * (ub[i] - lb[i]) + lb[i]
    convergence_curve = numpy.zeros(num_of_generations)

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(num_of_generations):
        ga = crossoverPopulaton(ga, scores, popSize, cp, keep, lb, ub,selection_type, crossover_type, mut_prob )
        mutatePopulaton(ga, popSize, mp, keep, lb, ub)
        ga = clearDups(ga, lb, ub)
        scores = calculateCost(objf, ga, popSize, lb, ub)
        bestScore = min(scores)
        ga, scores = sortPopulation(ga, scores)
        convergence_curve[l] = bestScore

        if l % 1 == 0:
            s.result.append([l+1,bestScore])

    timerEnd = time.time()
    s.bestIndividual = bestIndividual
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.best = bestScore
    s.convergence = convergence_curve
    s.optimizer = "GA"
    s.objfname = objf.__name__
    s.ga = ga
    return s

def read(objf, lb, ub, dim, popSize, num_of_generations, ga, mut_prob,selection_type):
    cp = 1  # crossover Probability
    mp = mut_prob  # Mutation Probability
    keep = 2

    s = solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    bestIndividual = numpy.zeros(dim)
    scores = numpy.random.uniform(0.0, 1.0, popSize)
    bestScore = float("inf")

    convergence_curve = numpy.zeros(num_of_generations)


    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(num_of_generations):
        ga = crossoverPopulaton(ga, scores, popSize, cp, keep, lb, ub,selection_type)
        mutatePopulaton(ga, popSize, mp, keep, lb, ub)
        ga = clearDups(ga, lb, ub)
        scores = calculateCost(objf, ga, popSize, lb, ub)
        bestScore = min(scores)
        ga, scores = sortPopulation(ga, scores)
        convergence_curve[l] = bestScore

        if l % 1 == 0:
            print(["At iteration " + str(l + 1) + " the best fitness is " + str(bestScore)])
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

if __name__ == "__main__":
    pass  # Add any testing or main code here if needed
