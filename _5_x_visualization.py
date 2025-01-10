import os
import argparse
import numpy as np

from collections import deque
from matplotlib import pyplot as plt
from _5_x_constants import (
    VISUALIZATION_FILE_EXTENSION,
    SINGLE_THREADED_GIL_KEY,
    MULTI_THREADED_GIL_KEY,
    MULTI_PROCESS_GIL_KEY,
    SINGLE_THREADED_KEY,
    MULTI_THREADED_KEY,
    MULTI_PROCESS_KEY,
    SINGLE_THREAD_PERFORMANCE_KEY,
    MULTI_THREAD_PERFORMANCE_KEY,
    MULTI_PROCESS_PERFORMANCE_KEY,
    KEYS,
    LABELS,
    COLORS,
)


# Input function for loading _5_x_script generated data from a visualization data format file
def load_visualization_data_from_file(
    file_path: str,
) -> tuple[str, list[tuple[int, float, float, float]]]:
    with open(file_path, "r") as f:
        title = f.readline().strip()
        if not title:
            raise ValueError("The visualization file must have a title.")
        data = [
            (int(values[0]), *map(float, values[1:]))
            for line in f
            if (values := line.strip().split())
        ]

    return (title, data)


# Output function for saving _5_x_script generated data to a visualization data format file
def save_visualization_data_to_file(
    file_path: str, data: str, mode: str = "a"
) -> None:
    # Check if the mode is valid
    if mode not in ["w", "a"]:
        raise ValueError(
            "Invalid mode. Use 'w' for overwrite or 'a' for append."
        )
    # Create the output directory if it does not exist
    output_directory = os.path.dirname(file_path)
    os.makedirs(output_directory, exist_ok=True)
    # Create the file path with the visualization file extension
    file_path = os.path.splitext(file_path)[0] + VISUALIZATION_FILE_EXTENSION
    # Save the data to the file
    with open(file_path, mode) as f:
        f.write(data + "\n")


# Calculate the minimum and maximum values for the execution times
def __execution_min_max(
    data: list[tuple[int, float, float, float]]
) -> tuple[int, int]:
    return (
        min(min(values[1], values[2], values[3]) for values in data),
        max(max(values[1], values[2], values[3]) for values in data),
    )


# Calculate the minimum and maximum values for the performance gains
def __performance_min_max(
    data: list[tuple[int, float, float, float]]
) -> tuple[int, int]:
    return (
        min(
            min(
                (data[i + 1][1] - data[i][1]) / data[i][1] * -100,
                (data[i + 1][2] - data[i][2]) / data[i][2] * -100,
                (data[i + 1][3] - data[i][3]) / data[i][3] * -100,
                -100,
            )
            for i in range(0, len(data), 2)
        ),
        max(
            max(
                (data[i + 1][1] - data[i][1]) / data[i][1] * -100,
                (data[i + 1][2] - data[i][2]) / data[i][2] * -100,
                (data[i + 1][3] - data[i][3]) / data[i][3] * -100,
                100,
            )
            for i in range(0, len(data), 2)
        ),
    )


# Main function
def main(file_path: str) -> None:
    # Load the visualization data from the file
    visualization = load_visualization_data_from_file(file_path)
    title, data = visualization

    # Initialize the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    formatter = plt.ScalarFormatter(useOffset=True, useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 4))
    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title)

    # First subplot
    ax1.set_title("Execution Times")
    ax1.set_xlabel("Load")
    ax1.set_ylabel("Time (s)")
    ax1.set_xlim(data[0][0], data[-1][0])
    ax1.set_ylim(__execution_min_max(data))
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(formatter)

    # Second subplot
    ax2.set_title("Performance Gains Without GIL")
    ax2.set_xlabel("Load")
    ax2.set_ylabel("Performance gain (%)")
    ax2.set_xlim(data[0][0], data[-1][0])
    _ymin, _ymax = __performance_min_max(data)
    ax2.set_ylim(_ymin, _ymax)
    ax2.grid(True)
    # Fill the areas
    x = [values[0] for values in data]
    ymin = [_ymin for _ in range(len(x))]
    ymid = [0 for _ in range(len(x))]
    ymax = [_ymax for _ in range(len(x))]
    ax2.fill_between(x, ymid, ymax, color='green', alpha=0.2)
    ax2.fill_between(x, ymin, ymid, color='red', alpha=0.2)
    ax2.xaxis.set_major_formatter(formatter)

    # Storage
    data_points = {key: deque(maxlen=len(data) // 2) for key in KEYS}
    scatter_points = {
        key: (
            ax2.scatter([], [], s=10, c=COLORS[key])
            if key
            in [
                SINGLE_THREAD_PERFORMANCE_KEY,
                MULTI_THREAD_PERFORMANCE_KEY,
                MULTI_PROCESS_PERFORMANCE_KEY,
            ]
            else ax1.scatter([], [], s=10, c=COLORS[key])
        )
        for key in KEYS
    }
    lines = {
        key: (
            ax2.plot([], [], label=LABELS[key], c=COLORS[key])[0]
            if key
            in [
                SINGLE_THREAD_PERFORMANCE_KEY,
                MULTI_THREAD_PERFORMANCE_KEY,
                MULTI_PROCESS_PERFORMANCE_KEY,
            ]
            else (
                ax1.plot([], [], label=LABELS[key], ls=":", c=COLORS[key])[0]
                if key
                in [
                    SINGLE_THREADED_GIL_KEY,
                    MULTI_THREADED_GIL_KEY,
                    MULTI_PROCESS_GIL_KEY,
                ]
                else ax1.plot([], [], label=LABELS[key], c=COLORS[key])[0]
            )
        )
        for key in KEYS
    }

    # Iterate through the data and update the plot
    for i in range(0, len(data), 2):
        values_gil = data[i]
        values = data[i + 1]

        for j, key in enumerate(data_points.keys()):
            if key in [
                SINGLE_THREADED_GIL_KEY,
                MULTI_THREADED_GIL_KEY,
                MULTI_PROCESS_GIL_KEY,
            ]:
                data_points[key].append((values_gil[0], values_gil[j + 1]))
                scatter_points[key].set_offsets(data_points[key])
                x = [x for x, y in data_points[key]]
                y = [y for x, y in data_points[key]]
                if i != 0:
                    coefficients = np.polyfit(x, y, 3, rcond=None)
                    trendline = np.poly1d(coefficients)
                    x_fit = np.linspace(min(x), max(x), 100)
                    y_fit = trendline(x_fit)
                    lines[key].set_data(x_fit, y_fit)
                else:
                    lines[key].set_data(x, y)
            elif key in [
                SINGLE_THREADED_KEY,
                MULTI_THREADED_KEY,
                MULTI_PROCESS_KEY,
            ]:
                data_points[key].append((values[0], values[j - 2]))
                scatter_points[key].set_offsets(data_points[key])
                x = [x for x, y in data_points[key]]
                y = [y for x, y in data_points[key]]
                if i != 0:
                    coefficients = np.polyfit(x, y, 3, rcond=None)
                    trendline = np.poly1d(coefficients)
                    x_fit = np.linspace(min(x), max(x), 100)
                    y_fit = trendline(x_fit)
                    lines[key].set_data(x_fit, y_fit)
                else:
                    lines[key].set_data(x, y)
            else:
                data_points[key].append(
                    (
                        values[0],
                        (values[j - 5] - values_gil[j - 5])
                        / values_gil[j - 5]
                        * -100,
                    )
                )
                scatter_points[key].set_offsets(data_points[key])
                x = [x for x, y in data_points[key]]
                y = [y for x, y in data_points[key]]
                if i != 0:
                    coefficients = np.polyfit(x, y, 3, rcond=None)
                    trendline = np.poly1d(coefficients)
                    x_fit = np.linspace(min(x), max(x), 100)
                    y_fit = trendline(x_fit)
                    lines[key].set_data(x_fit, y_fit)
                else:
                    lines[key].set_data(x, y)
        plt.tight_layout()
        plt.pause(0.1)

    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel Processing in Python: Visualization"
    )

    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the file containing visualization data .vis",
    )

    args = parser.parse_args()

    # Run the main function
    main(
        file_path=args.file_path,
    )
