import numpy as np
from typing import Literal


def generate_neighbors(x, epsilon, maxneighbors, max_x_limit, min_x_limit):
    x = np.array(x).reshape(2, 1)
    random_perturbations = np.random.uniform(-epsilon, +epsilon, maxneighbors).reshape(
        maxneighbors, 1
    )
    candidates = np.empty((maxneighbors, 2))
    candidates[:, 0] = x[0] + random_perturbations[:, 0]
    candidates[:, 1] = x[1] + random_perturbations[:, 0]

    clippedcandidates = np.clip(
        candidates,
        a_min=min_x_limit,
        a_max=max_x_limit,
    )
    return clippedcandidates.T


def hill_climbing(
    f,
    epsilon,
    maxiterations,
    maxneighbors,
    type: Literal["max", "min"],
    max_x_limit=[100, 100],
    min_x_limit=[-100, -100],
    max_not_improved_iterations=5,
):
    if type not in ["max", "min"]:
        raise ValueError("type must be either 'max' or 'min'")

    x_best = min_x_limit
    f_best = f(x_best)
    i = 0
    improvement = max_not_improved_iterations

    while i < maxiterations:
        improvement -= 1

        y = generate_neighbors(x_best, epsilon, maxneighbors, max_x_limit, min_x_limit)
        F = f(y)
        best_neighbor_index = np.argmax(F) if type == "max" else np.argmin(F)
        y = y[:, best_neighbor_index]
        new_best_f = F[best_neighbor_index]

        if (type == "max" and new_best_f > f_best) or (
            type == "min" and new_best_f < f_best
        ):
            x_best, f_best = y, new_best_f
            improvement = max_not_improved_iterations

        i += 1

        if not improvement:
            print("Early stopping Hill Climbing due to no improvement")
            break

    return np.array(x_best)
