import matplotlib.pyplot as plt


def plot_best_path(points, best_paths, best_distances):
    # Number of paths to plot
    num_paths = len(best_paths)

    for path_index in range(num_paths):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Extract path and distance for the current plot
        current_path = best_paths[path_index].astype(int)
        current_distance = best_distances[path_index]

        # Extracting x, y, z coordinates of the current path
        x, y, z = (
            points[current_path][:, 0],
            points[current_path][:, 1],
            points[current_path][:, 2],
        )

        # Plotting the path
        for i in range(len(current_path) - 1):
            ax.plot(
                x[i : i + 2],
                y[i : i + 2],
                z[i : i + 2],
                color=plt.cm.viridis(i / (len(current_path) - 1)),
                marker="o",
            )

        # Highlighting start and end points
        ax.plot([x[0]], [y[0]], [z[0]], "go")  # Start point in green
        ax.plot([x[-1]], [y[-1]], [z[-1]], "ro")  # End point in red

        # Annotating points with their sequence in the path
        for i, (px, py, pz) in enumerate(zip(x, y, z)):
            ax.text(px, py, pz, f"{i}", color="black", fontsize=8)

        # Setting labels and title
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_title(
            f"Path {path_index + 1} in 3D with Total Distance: {current_distance:.2f}"
        )

        plt.show()
