import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import plot_best_path

num_points = 10
N = 100
max_generations = 1000
crossover_probability = 0.9
mutation_probability = 0.05
tournament_size = 5
elite_size = 1

best_stored = 2

xmin, xmax, ymin, ymax, zmin, zmax = 0, 100, 0, 100, 0, 100

points = np.random.rand(num_points, 3)
points *= np.array([[xmax - xmin, ymax - ymin, zmax - zmin]])
points += np.array([[xmin, ymin, zmin]])


def fitness(individual):
    dist = np.sum(np.sqrt(np.sum(np.diff(points[individual], axis=0) ** 2, axis=1)))
    return dist + np.sqrt(np.sum((points[individual[0]] - points[individual[-1]]) ** 2))


def initialize_population():
    return [np.random.permutation(num_points) for _ in range(N)]


def tournament_selection(population, fitness_values):
    selection_ix = np.random.randint(len(population))

    for ix in np.random.randint(0, len(population), tournament_size - 1):
        if fitness_values[ix] < fitness_values[selection_ix]:
            selection_ix = ix
    return population[selection_ix]


def fill_child(child, parent):
    for i in range(num_points):
        if child[i] is None:
            for p_idx in range(len(parent)):
                if parent[p_idx] not in child:
                    child[i] = parent[p_idx]
                    break


def two_point_crossover(parent1, parent2):
    if np.random.rand() > crossover_probability:
        return parent1, parent2

    cp1, cp2 = np.sort(np.random.choice(num_points, 2, replace=False))

    child1 = [None for _ in range(num_points)]
    child2 = [None for _ in range(num_points)]

    child1[cp1:cp2] = parent1[cp1:cp2]
    child2[cp1:cp2] = parent2[cp1:cp2]

    fill_child(child1, parent2)
    fill_child(child2, parent1)

    return np.array(child1), np.array(child2)


def mutate(chromosome):
    if np.random.rand() < mutation_probability:
        mutation_points = np.random.randint(0, num_points, size=2)
        chromosome[mutation_points[0]], chromosome[mutation_points[1]] = (
            chromosome[mutation_points[1]],
            chromosome[mutation_points[0]],
        )
    return chromosome


def genetic_algorithm():
    population = initialize_population()

    best_distance = np.empty(best_stored)
    best_distance.fill(np.inf)

    best_path = np.empty((best_stored, num_points))
    scores = np.zeros(N)

    for generation in range(max_generations):
        for i in range(N):
            scores[i] = fitness(population[i])
            if scores[i] < best_distance.min():
                best_distance = np.insert(best_distance, 0, scores[i])[0:-1]
                best_path = np.insert(best_path, 0, population[i], axis=0)[0:-1]

        children = []
        for i in range(0, N, 2):
            selected_parent1 = tournament_selection(population, scores)
            selected_parent2 = tournament_selection(population, scores)

            child1, child2 = two_point_crossover(selected_parent1, selected_parent2)

            children.append(mutate(child1))
            children.append(mutate(child2))

        elite_indices = scores.argsort()[:elite_size]
        elite_individuals = [population[index] for index in elite_indices]
        population = elite_individuals + children[: N - elite_size]

        if generation % 100 == 0:
            print(f"Generation {generation}, Best distances: {best_distance}")

    return best_path, best_distance


best_paths, best_distances = genetic_algorithm()

plot_best_path(points, best_paths, best_distances)
