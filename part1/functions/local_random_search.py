import numpy as np
from typing import Literal


def local_random_search(
    f,
    sd,
    maxiterations,
    type: Literal["max", "min"],
    max_x_limit=[100, 100],
    min_x_limit=[-100, -100],
    max_not_improved_iterations=5,
):
    if type not in ["max", "min"]:
        raise ValueError("type must be either 'max' or 'min'")

    x_best = np.random.uniform(min_x_limit, max_x_limit)
    f_best = f(x_best)
    i = 0
    improvement = max_not_improved_iterations

    while i < maxiterations:
        improvement -= 1

        n = np.random.normal(0, sd)
        x_candidate = x_best + n
        x_candidate = np.clip(x_candidate, min_x_limit, max_x_limit)
        F = f(x_candidate)

        if (type == "max" and F > f_best) or (type == "min" and F < f_best):
            x_best, f_best = x_candidate, F
            improvement = max_not_improved_iterations

        i += 1

        if not improvement:
            print(f"Early stopping Local Random Search due to no improvement {x_best}")
            break

    return x_best
