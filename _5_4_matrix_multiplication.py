# Script (_5_4_matrix_multiplication.py):
# A demonstration of parallel processing for CPU-bound task in Python 3.13.
# This script explores concurrent performance gains and comparisons between threading with and without GIL.
#
# Prerequisites:
# - Python 3.13.1 with experimental free-threading installed in the system environment.
# - Run with GIL:
#       ~/AppData/Local/Programs/Python/Python313/python.exe _5_4_matrix_multiplication.py ./data_results/_5_4/matrix_multiplication_sample_a.txt ./data_results/_5_4/matrix_multiplication_sample_b.txt ./data_results/_5_4
# - Run without GIL:
#       ~/AppData/Local/Programs/Python/Python313/python3.13t.exe _5_4_matrix_multiplication.py ./data_results/_5_4/matrix_multiplication_sample_a.txt ./data_results/_5_4/matrix_multiplication_sample_b.txt ./data_results/_5_4
#
# Results Summary (matrix_a dimen 543x543 and matrix_b dimen 543x777, using 4 threads and 4 processes):
# - With GIL:
#   * Single threaded: completed in 12.98 seconds.
#   * Multi threaded: completed in 13.52 seconds.
#   * Multi process: completed in 3.89 seconds.
# - Without GIL:
#   * Single threaded: completed in 19.26 seconds.
#   * Multi threaded: completed in 5.71 seconds.
#   * Multi process: completed in 5.87 seconds.
#
# Visualized Results (supported by Python 3.12):
# - Use the visualization script to generate a graph of the results.
# - Run: python _5_x_visualization.py ./data_results/visualizations/matrix_multiplication.vis
#
# Note:
# - Single-threaded performance is worse without GIL as the load increases.
# - Multi-threaded performance benefits significantly from the absence of GIL as the load increases.
# - Multi-process execution performance decreases significantly but performs around the same as multi-threaded without GIL.
#
# Paper: Parallel Processing – An In-Depth Look Into Python 3.13 (2025)
# Authors: Mantvydas Deltuva and Justinas Teselis

import os
import sys
import time
import random
import logging
import argparse
import multiprocessing
import concurrent.futures

from _5_x_constants import (
    NUM_THREADS,
    NUM_PROCESSES,
    LOG_LEVEL,
    MATRIX_SINGLE_THREADED_RESULT_FILE_NAME,
    MATRIX_MULTI_THREADED_RESULT_FILE_NAME,
    MATRIX_MULTI_PROCESS_RESULT_FILE_NAME,
)

logger = logging.getLogger(__name__)


# A CPU-bound task: matrix multiplication of row and column
def __row_col_multiplication(row: list[int], col: list[int]) -> int:
    # Inner product of two vectors
    return sum(a * b for a, b in zip(row, col))


# Single Threaded (consecutive)
def matrix_multiplication(
    matrix_a: list[list[int]], matrix_b: list[list[int]]
) -> list[list[int]]:
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Matrix dimensions do not match.")

    matrix_b_transposed = list(zip(*matrix_b))

    # Matrix multiplication
    return [
        [__row_col_multiplication(row, col) for col in matrix_b_transposed]
        for row in matrix_a
    ]


# Multi Threaded (concurrent)
def threaded_matrix_multiplication(
    matrix_a: list[list[int]], matrix_b: list[list[int]], num_threads: int
) -> list[list[int]]:
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Matrix dimensions do not match.")

    # Worker function
    def __worker(i: int) -> list[int]:
        intermediate_row = []
        for j in range(len(matrix_b_transposed)):
            intermediate_row.append(
                __row_col_multiplication(matrix_a[i], matrix_b_transposed[j])
            )
        return intermediate_row

    matrix_b_transposed = list(zip(*matrix_b))
    result = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
        for i in range(len(matrix_a)):
            result.append(executor.submit(__worker, i))

    concurrent.futures.wait(result)
    return [future.result() for future in result]


# Multi Process (concurrent) worker function
def __process_worker(
    i: int,
    matrix_a: list[list[int]],
    matrix_b_transposed: list[list[int]],
) -> list[int]:
    intermediate_row = []
    for j in range(len(matrix_b_transposed)):
        intermediate_row.append(
            __row_col_multiplication(matrix_a[i], matrix_b_transposed[j])
        )
    return intermediate_row


# Multi Process (concurrent)
def process_matrix_multiplication(
    matrix_a: list[list[int]], matrix_b: list[list[int]], num_processes: int
) -> list[list[int]]:
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Matrix dimensions do not match.")

    matrix_b_transposed = list(zip(*matrix_b))
    tasks = [
        (
            # rows index
            i,
            # first matrix
            matrix_a,
            # second matrix transposed
            matrix_b_transposed,
        )
        for i in range(len(matrix_a))
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        result = pool.starmap(__process_worker, tasks)

    return result


# Input function
def load_data_from_file(file_path: str) -> list[list[int]]:
    with open(file_path, "r") as f:
        rows, cols = map(int, f.readline().strip().split())
        if rows <= 0 or cols <= 0:
            raise ValueError(
                f"Invalid matrix dimensions: {rows}x{cols}. Both rows and columns must be greater than zero."
            )
        matrix = [list(map(int, line.strip().split())) for line in f]
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            raise ValueError("Matrix dimensions do not match the data.")

    return matrix


# Output function
def save_data_to_file(matrix: list[list[int]], file_path: str) -> None:
    with open(file_path, "w") as f:
        rows = len(matrix)
        cols = len(matrix[0])
        f.write(f"{rows} {cols}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")


# Matrix generation function
def generate_matrix(rows: int, cols: int) -> list[list[int]]:
    return [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]


# Main function
def main(
    matrix_a_file_path: str,
    matrix_b_file_path: str,
    output_folder_path: str = None,
    num_threads: int = NUM_THREADS,
    num_processes: int = NUM_PROCESSES,
    log_level: str = LOG_LEVEL,
) -> tuple[int, float, float, float]:
    # Logging configuration (default: debug)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.DEBUG)
    )

    # Flags
    OUTPUT = output_folder_path is not None

    # Create the output folder if it does not exist
    if OUTPUT:
        os.makedirs(output_folder_path, exist_ok=True)

    # Variables
    matrix_a = load_data_from_file(matrix_a_file_path)
    matrix_b = load_data_from_file(matrix_b_file_path)

    # Feedback of the program status
    logger.debug(
        f"matrix_a dimen {f"{len(matrix_a)}x{len(matrix_a[0])}"} and "
        + f"matrix_b dimen {f"{len(matrix_b)}x{len(matrix_b[0])}"}, "
        + f"using {num_threads} threads and {num_processes} processes"
    )
    logger.debug(f"The GIL is active: {sys._is_gil_enabled()}")

    # -----------------------------
    # Single Threaded (consecutive)
    # -----------------------------

    # Matrix multiplication
    start_time = time.time()
    single_threaded_result = matrix_multiplication(matrix_a, matrix_b)
    single_threaded_time = time.time() - start_time

    # Feedback of single threaded execution time
    logger.debug(
        f"Single threaded: matrix multiplication completed in {single_threaded_time:.2f} seconds"
    )

    # Save single threaded data to a file
    if OUTPUT:
        output_single_threaded_file = os.path.join(
            output_folder_path,
            MATRIX_SINGLE_THREADED_RESULT_FILE_NAME,
        )

        save_data_to_file(single_threaded_result, output_single_threaded_file)

    # ---------------------------
    # Multi Threaded (concurrent)
    # ---------------------------

    # Matrix multiplication
    start_time = time.time()
    multi_threaded_result = threaded_matrix_multiplication(
        matrix_a, matrix_b, num_threads
    )
    multi_threaded_time = time.time() - start_time

    # Feedback of multi threaded execution time
    logger.debug(
        f"Multi threaded: matrix multiplication completed in {multi_threaded_time:.2f} seconds"
    )

    # Save multi threaded data to a file
    if OUTPUT:
        output_multi_threaded_file = os.path.join(
            output_folder_path, MATRIX_MULTI_THREADED_RESULT_FILE_NAME
        )
        save_data_to_file(multi_threaded_result, output_multi_threaded_file)

    # --------------------------
    # Multi Process (concurrent)
    # --------------------------

    # Matrix multiplication
    start_time = time.time()
    multi_process_result = process_matrix_multiplication(
        matrix_a, matrix_b, num_processes
    )
    multi_process_time = time.time() - start_time

    # Feedback of multi process execution time
    logger.debug(
        f"Multi process: matrix multiplication completed in completed in {multi_process_time:.2f} seconds"
    )

    # Save multi process data to a file
    if OUTPUT:
        output_multi_process_file = os.path.join(
            output_folder_path, MATRIX_MULTI_PROCESS_RESULT_FILE_NAME
        )
        save_data_to_file(multi_process_result, output_multi_process_file)

    # STDOUT
    print(
        f"{len(matrix_a) * len(matrix_a[0]) * len(matrix_b[0])} {single_threaded_time} {multi_threaded_time} {multi_process_time}"
    )

    # Returns a tuple:
    # 1. The total number of multiplications done during the matrix multiplication.
    # 2. The execution time for the single-threaded matrix multiplication.
    # 3. The execution time for the multi-threaded matrix multiplication.
    # 4. The execution time for the multi-process matrix multiplication.
    return (
        len(matrix_a) * len(matrix_a[0]) * len(matrix_b[0]),
        single_threaded_time,
        multi_threaded_time,
        multi_process_time,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel Processing in Python: Matrix Multiplication"
    )

    parser.add_argument(
        "matrix_a_file_path",
        type=str,
        help="Path to the file containing matrix A",
    )
    parser.add_argument(
        "matrix_b_file_path",
        type=str,
        help="Path to the file containing matrix B",
    )
    parser.add_argument(
        "output_folder_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the output folder for matrix multiplication results (optional)",
    )
    parser.add_argument(
        "-t",
        "--num_threads",
        type=int,
        default=NUM_THREADS,
        help="Number of threads to use (default: 4)",
    )
    parser.add_argument(
        "-p",
        "--num_processes",
        type=int,
        default=NUM_PROCESSES,
        help="Number of processes to use (default: 4)",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default=LOG_LEVEL,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level (default: debug)",
    )

    args = parser.parse_args()

    # Run the main function
    main(
        matrix_a_file_path=args.matrix_a_file_path,
        matrix_b_file_path=args.matrix_b_file_path,
        output_folder_path=args.output_folder_path,
        num_threads=args.num_threads,
        num_processes=args.num_processes,
        log_level=args.log_level,
    )
