import numpy as np
from typing import Literal


def global_random_search(
    f,
    maxiterations,
    type: Literal["max", "min"],
    max_x_limit=[100, 100],
    min_x_limit=[-100, -100],
):
    if type not in ["max", "min"]:
        raise ValueError("type must be either 'max' or 'min'")

    x_best = np.random.uniform(min_x_limit, max_x_limit)
    f_best = f(x_best)
    i = 0

    while i < maxiterations:
        n = np.random.uniform(min_x_limit, max_x_limit)
        x_candidate = x_best + n
        x_candidate = np.clip(x_candidate, min_x_limit, max_x_limit)
        F = f(x_candidate)

        if (type == "max" and F > f_best) or (type == "min" and F < f_best):
            x_best, f_best = x_candidate, F

        i += 1

    return x_best
