from typing import Literal
import numpy as np
from functions.global_random_search import global_random_search
from functions.hill_climbing import hill_climbing
from functions.local_random_search import local_random_search
from functions.simulated_annealing import simulated_annealing
from matplotlib import pyplot as plt

from utils import plot_3d_graph, plot_modes


def optimize(
    f,
    epsilon=0.01,
    maxiterations=1000,
    maxneighbors=10,
    sd=0.5,
    T=800,
    type: Literal["max", "min"] = "min",
    max_x_limit=[100, 100],
    min_x_limit=[-100, -100],
    plot=False,
    max_not_improved_iterations=5,
):
    i = 0
    best_x_values = np.empty((rounds, 4, 2))

    if plot:
        plot_3d_graph(
            [[min_x_limit[0], max_x_limit[0]], [min_x_limit[1], max_x_limit[1]]], f
        )

    while i < rounds:
        best_x_hc = hill_climbing(
            f,
            epsilon,
            maxiterations,
            maxneighbors,
            type,
            max_x_limit,
            min_x_limit,
            max_not_improved_iterations,
        )

        best_x_lrs = local_random_search(
            f,
            sd,
            maxiterations,
            type,
            max_x_limit,
            min_x_limit,
            max_not_improved_iterations,
        )

        best_x_grs = global_random_search(
            f, maxiterations, type, max_x_limit, min_x_limit
        )

        best_x_sa = simulated_annealing(
            f,
            sd,
            T,
            maxiterations,
            type,
            max_x_limit,
            min_x_limit,
        )

        best_round_x_values = np.empty((0, 2))
        best_round_x_values = np.concatenate(
            (best_round_x_values, best_x_hc.reshape(1, 2)),
        )
        best_round_x_values = np.concatenate(
            (best_round_x_values, best_x_lrs.reshape(1, 2)),
        )
        best_round_x_values = np.concatenate(
            (best_round_x_values, best_x_grs.reshape(1, 2)),
        )
        best_round_x_values = np.concatenate(
            (best_round_x_values, best_x_sa.reshape(1, 2)),
        )

        best_x_values[i] = best_round_x_values

        i += 1

    return best_x_values


rounds = 100


epsilon = 0.5
maxiterations = 1000
maxneighbors = 10

sd = 10
T = 800

type = "min"


max_x_limit = [100, 100]
min_x_limit = [-100, -100]
plot = False
max_not_improved_iterations = 100


# def f_1(x):
#     return x[0] ** 2 + x[1] ** 2


# y_1 = optimize(
#     f_1,
#     epsilon,
#     maxiterations,
#     maxneighbors,
#     sd,
#     T,
#     type,
#     max_x_limit,
#     min_x_limit,
#     plot,
#     max_not_improved_iterations,
# )

# plot_modes(y_1, f_1, plot_mean=True)

# ======================================================================================================

# def f_2(x):
#     return np.exp(-(x[0] ** 2 + x[1] ** 2)) + 2 * np.exp(
#         -((x[0] - 1.7) ** 2 + (x[1] - 1.7) ** 2)
#     )


# max_x_limit = [4, 5]
# min_x_limit = [-2, -2]
# T = 1200
# type = "max"

# y_2 = optimize(
#     f_2,
#     epsilon,
#     maxiterations,
#     maxneighbors,
#     sd,
#     T,
#     type,
#     max_x_limit,
#     min_x_limit,
#     plot,
#     max_not_improved_iterations,
# )

# plot_modes(y_2, f_2)

# ======================================================================================================


# def f_3(x):
#     return (
#         -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
#         - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
#         + 20
#         + np.e**1
#     )


# type = "min"

# max_x_limit = [8, 8]
# min_x_limit = [-8, -8]

# plot = True

# y_3 = optimize(
#     f_3,
#     epsilon,
#     maxiterations,
#     maxneighbors,
#     sd,
#     T,
#     type,
#     max_x_limit,
#     min_x_limit,
#     plot,
#     max_not_improved_iterations,
# )

# plot_modes(y_3, f_3)


# ======================================================================================================


# def f_4(x):
#     return (x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0]) + 10) + (
#         x[1] ** 2 - 10 * np.cos(2 * np.pi * x[1]) + 10
#     )


# min_x_limit = [-5.12, -5.12]
# max_x_limit = [5.12, 5.12]

# type = "min"

# plot = True

# y_4 = optimize(
#     f_4,
#     epsilon,
#     maxiterations,
#     maxneighbors,
#     sd,
#     T,
#     type,
#     max_x_limit,
#     min_x_limit,
#     plot,
#     max_not_improved_iterations,
# )

# plot_modes(y_4, f_4)

# ======================================================================================================


# def f_5(x):
#     return (x[0] - 1) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# min_x_limit = [-2, -1]
# max_x_limit = [2, 3]

# type = "min"

# plot = True

# y_5 = optimize(
#     f_5,
#     epsilon,
#     maxiterations,
#     maxneighbors,
#     sd,
#     T,
#     type,
#     max_x_limit,
#     min_x_limit,
#     plot,
#     max_not_improved_iterations,
# )

# plot_modes(y_5, f_5)

# ======================================================================================================


# def f_6(x):
#     return x[0] * np.sin(4 * np.pi * x[0]) - x[1] * np.sin(4 * np.pi * x[1] + np.pi) + 1


# max_x_limit = [3, 3]
# min_x_limit = [-1, -1]

# type = "max"

# plot = True

# y_6 = optimize(
#     f_6,
#     epsilon,
#     maxiterations,
#     maxneighbors,
#     sd,
#     T,
#     type,
#     max_x_limit,
#     min_x_limit,
#     plot,
#     max_not_improved_iterations,
# )

# plot_modes(y_6, f_6)


# ======================================================================================================


def f_7(x):
    return -np.sin(x[0]) * np.sin((x[0] ** 2 / np.pi)) ** (2 * 10) - np.sin(
        x[1]
    ) * np.sin((2 * x[1] ** 2 / np.pi)) ** (2 * 10)


max_x_limit = [np.pi, np.pi]
min_x_limit = [0, 0]

type = "min"

plot = True

y_7 = optimize(
    f_7,
    epsilon,
    maxiterations,
    maxneighbors,
    sd,
    T,
    type,
    max_x_limit,
    min_x_limit,
    plot,
    max_not_improved_iterations,
)


plot_modes(y_7, f_7)

# ======================================================================================================


# def f_8(x):
#     return -(x[1] + 47) * np.sin(np.sqrt(abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(
#         np.sqrt(abs(x[0] - (x[1] + 47)))
#     )


# max_x_limit = [20, 20]
# min_x_limit = [-200, -200]

# type = "min"

# plot = True

# y_8 = optimize(
#     f_8,
#     epsilon,
#     maxiterations,
#     maxneighbors,
#     sd,
#     T,
#     type,
#     max_x_limit,
#     min_x_limit,
#     plot,
#     max_not_improved_iterations,
# )

# plot_modes(y_8, f_8)
