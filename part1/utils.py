from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_modes(data, f, plot_mean=False, float_precision=2, sizing=0.2):
    labels = [
        "Hill Climbing",
        "Local Random Search",
        "Global Random Search",
        "Simulated Annealing",
    ]

    data = np.around(data, float_precision)

    _, n_labels, _ = data.shape

    mode_data = np.zeros((n_labels, 2))
    mean_data = np.zeros((n_labels, 2))

    for j in range(2):
        for i in range(n_labels):
            mode_result = stats.mode(data[:, i, j])
            mode_data[i, j] = mode_result.mode
            mean_data[i, j] = np.mean(data[:, i, j])

    y_mode = np.around(f(mode_data.T), float_precision).reshape(4, 1)
    mode_data_with_f = np.concatenate((mode_data, y_mode), axis=1)

    y_mean = np.around(f(mean_data.T), float_precision).reshape(4, 1)
    mean_data_with_f = np.concatenate((mean_data, y_mean), axis=1)

    mode_and_mean_data = mode_data_with_f

    colLabels = ["Mode(X1)", "Mode(X2)", "F(Mode(X))"]

    if plot_mean:
        mode_and_mean_data = np.concatenate(
            (mode_data_with_f, mean_data_with_f), axis=1
        )
        colLabels = [
            "Mode(X1)",
            "Mode(X2)",
            "F(Mode(X))",
            "Mean(X1)",
            "Mean(X2)",
            "F(Mean(X))",
        ]

    _, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")
    ax.table(
        cellText=np.trunc(mode_and_mean_data * 10**float_precision)
        / (10**float_precision),
        rowLabels=labels,
        colLabels=colLabels,
        loc="center",
        colWidths=[sizing for x in mode_and_mean_data[0]],
    )

    plt.show()


def plot_3d_graph(limits, f):
    x1_limit, x2_limit = limits
    x = np.linspace(x1_limit[0], x1_limit[1], 100)
    y = np.linspace(x2_limit[0], x2_limit[1], 100)
    X, Y = np.meshgrid(x, y)

    Z = f([X, Y])

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_title("Surface plot")
    plt.show()
