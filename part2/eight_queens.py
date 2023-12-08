import numpy as np
import matplotlib.pyplot as plt

N = 100
max_generations = 1000
crossover_probability = 0.9
mutation_probability = 0.05
n_queens = 8

possible_solutions = np.empty((0, n_queens))
genetic_algorithm_runs = 0


def fitness(chromosome):
    non_attacking_pairs = 28
    for i in range(n_queens):
        for j in range(i + 1, n_queens):
            if (
                chromosome[i] == chromosome[j]
                or abs(chromosome[i] - chromosome[j]) == j - i
            ):
                non_attacking_pairs -= 1
    return non_attacking_pairs


def initialize_population():
    return np.array([np.random.permutation(n_queens) for _ in range(N)])


def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = fitness_values / total_fitness
    selected_parent = population[np.random.choice(range(N), size=1, p=selection_probs)]
    return selected_parent[0]


def one_point_crossover(parent1, parent2):
    if np.random.rand() > crossover_probability:
        return parent1, parent2

    crossover_point = np.random.randint(1, n_queens)
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2


def mutate(chromosome):
    if np.random.rand() < mutation_probability:
        mutation_points = np.random.randint(0, n_queens, size=2)
        chromosome[mutation_points[0]], chromosome[mutation_points[1]] = (
            chromosome[mutation_points[1]],
            chromosome[mutation_points[0]],
        )
    return chromosome


def check_circular_shifts(chromosome):
    circular_shifts = []

    for i in range(n_queens):
        circular_shifts.append(np.roll(chromosome, i))

    fit_individuals = []

    for individual in circular_shifts:
        if fitness(individual) == 28:
            fit_individuals.append(individual)
    return fit_individuals


def genetic_algorithm():
    global genetic_algorithm_runs

    population = initialize_population()
    best_fitness = 0
    best_solution = np.zeros(n_queens)
    generation = 0
    genetic_algorithm_runs += 1

    while generation < max_generations:
        fitness_values = np.array([fitness(indv) for indv in population])

        best_fitness_current_generation = np.max(fitness_values)
        best_individual_current_generation = population[np.argmax(fitness_values)]

        if best_fitness_current_generation > best_fitness:
            best_fitness = best_fitness_current_generation
            best_solution = best_individual_current_generation

        if best_fitness == 28:
            global possible_solutions
            circular_fit_individuals = check_circular_shifts(best_solution)

            for individual in circular_fit_individuals:
                if not np.any(np.all(possible_solutions == individual, axis=1)):
                    possible_solutions = np.concatenate(
                        (possible_solutions, individual.reshape(1, n_queens)),
                    )
                    print(
                        "Solution found: ",
                        individual,
                        " in run #",
                        genetic_algorithm_runs,
                    )

            break

        children = []
        for _ in range(0, N, 2):
            selected_parent1 = roulette_wheel_selection(population, fitness_values)
            selected_parent2 = roulette_wheel_selection(population, fitness_values)

            child1, child2 = one_point_crossover(selected_parent1, selected_parent2)
            children.append(mutate(child1))
            children.append(mutate(child2))

        children[0] = best_individual_current_generation  # elitism

        population = np.array(children)
        generation += 1


while possible_solutions.size < 10 * n_queens:
    genetic_algorithm()

print(possible_solutions)
