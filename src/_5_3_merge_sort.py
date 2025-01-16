# Script (_5_3_merge_sort.py):
# A demonstration of parallel processing for CPU-bound task in Python 3.13.
# This script explores concurrent performance gains and comparisons between threading with and without GIL.
#
# Prerequisites:
# - Python 3.13.1 with experimental free-threading installed in the system environment.
# - Run with GIL:
#       ~/AppData/Local/Programs/Python/Python313/python.exe _5_3_merge_sort.py ./data_results/_5_3/merge_sort_sample_1m.txt ./data_results/_5_3
# - Run without GIL:
#       ~/AppData/Local/Programs/Python/Python313/python3.13t.exe _5_3_merge_sort.py ./data_results/_5_3/merge_sort_sample_1m.txt ./data_results/_5_3
#
# Results Summary (1000000 numbers, using 4 threads and 4 processes):
# - With GIL:
#   * Single threaded: completed in 3.83 seconds.
#   * Multi threaded: completed in 3.92 seconds.
#   * Multi process: completed in 1.64 seconds.
# - Without GIL:
#   * Single threaded: completed in 7.20 seconds.
#   * Multi threaded: completed in 3.84 seconds.
#   * Multi process: completed in 3.07 seconds.
#
# Visualized Results (supported by Python 3.12):
# - Use the visualization script to generate a graph of the results.
# - Run: python _5_x_visualization.py ./data_results/visualizations/merge_sort.vis
# Note:
# - Single-threaded performance decreases drasticly without GIL.
# - Multi-threaded performance benefits are minor or non-existing from the absence of GIL.
# - Multi-process performance decreases without GIL.
#
# Paper: Parallel Processing – An In-Depth Look Into Python 3.13 (2025)
# Authors: Mantvydas Deltuva and Justinas Teselis

import os
import sys
import time
import random
import logging
import argparse
import threading
import multiprocessing

from _5_x_constants import (
    NUM_THREADS,
    NUM_PROCESSES,
    LOG_LEVEL,
    MERGE_SINGLE_THREADED_RESULT_FILE_NAME,
    MERGE_MULTI_THREADED_RESULT_FILE_NAME,
    MERGE_MULTI_PROCESS_RESULT_FILE_NAME,
)

logger = logging.getLogger(__name__)


# A CPU-bound task: merging two sorted lists
def merge(left: list[int], right: list[int]) -> list[int]:
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


# Single Threaded (consecutive)
def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr

    # Recursively split the array into halves
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    left = merge_sort(left)
    right = merge_sort(right)

    # Merge the sorted halves
    return merge(left, right)


# Multi Threaded (concurrent)
def threaded_merge_sort(arr: list[int], num_threads: int) -> list[int]:
    if len(arr) <= 1:
        return arr

    # Worker function
    def worker(i: int, start: int, end: int) -> None:
        results[i] = merge_sort(arr[start:end])

    # Step 1: Divide the array into chunks
    chunk_size = len(arr) // num_threads
    threads = []
    results = [None] * num_threads

    # Step 2: Sort chunks in parallel
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_threads - 1 else len(arr)
        thread = threading.Thread(target=worker, args=(i, start, end))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Step 3: Merge sorted chunks sequentially
    while len(results) > 1:
        next_round = []
        for i in range(0, len(results), 2):
            if i + 1 < len(results):
                next_round.append(merge(results[i], results[i + 1]))
            else:
                next_round.append(results[i])
        results = next_round

    return results[0]


# Multi Process (concurrent)
def process_merge_sort(arr: list[int], num_processes: int) -> list[int]:
    if len(arr) <= 1:
        return arr

    # Step 1: Divide the array into chunks
    chunk_size = len(arr) // num_processes
    chunks = [arr[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes - 1)]
    chunks.append(arr[(num_processes - 1) * chunk_size:])  # Add remaining elements
    
    # Step 2: Sort chunks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(merge_sort, chunks)

    # Step 3: Merge sorted chunks sequentially
    while len(results) > 1:
        next_round = []
        for i in range(0, len(results), 2):
            if i + 1 < len(results):
                next_round.append(merge(results[i], results[i + 1]))
            else:
                next_round.append(results[i])
        results = next_round

    return results[0]

# Input function
def load_data_from_file(file_path: str) -> list[int]:
    with open(file_path, "r") as f:
        return [int(line.strip()) for line in f]


# Output function
def save_data_to_file(arr: list[int], file_path: str) -> None:
    with open(file_path, "w") as f:
        for num in arr:
            f.write(f"{num}\n")

# Unsorted array generation function
def generate_array(n: int) -> list[int]:
    return [random.randint(1, n) for _ in range(n)]

# Main function
def main(
    array_file_path: str,
    output_folder_path: str = None,
    num_threads: int = NUM_THREADS,
    num_processes: int = NUM_PROCESSES,
    log_level: str = LOG_LEVEL,
):
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
    arr = load_data_from_file(array_file_path)

    # Feedback of the program status
    logger.debug(
        f"{len(arr)} numbers, using {num_threads} threads and {num_processes} processes"
    )
    logger.debug(f"The GIL is active: {sys._is_gil_enabled()}")

    # ------------------------------
    # Single Threaded (consecutive)
    # -----------------------------

    # Merge sort the array
    start_time = time.time()
    single_threaded_result = merge_sort(arr)
    single_threaded_time = time.time() - start_time

    # Feedback of single threaded execution time
    logger.debug(
        f"Single threaded: merge sort completed in {single_threaded_time:.2f} seconds"
    )

    # Save single threaded data to a file
    if OUTPUT:
        output_single_threaded_file = os.path.join(
            output_folder_path,
            MERGE_SINGLE_THREADED_RESULT_FILE_NAME,
        )

        save_data_to_file(single_threaded_result, output_single_threaded_file)

    # ----------------------------
    # Multi Threaded (concurrent)
    # ---------------------------

    # Merge sort the array
    start_time = time.time()
    multi_threaded_result = threaded_merge_sort(arr, num_threads)
    multi_threaded_time = time.time() - start_time

    # Feedback of multi threaded execution time
    logger.debug(
        f"Multi threaded: merge sort completed in {multi_threaded_time:.2f} seconds"
    )

    # Save multi threaded data to a file
    if OUTPUT:
        output_multi_threaded_file = os.path.join(
            output_folder_path, MERGE_MULTI_THREADED_RESULT_FILE_NAME
        )
        save_data_to_file(multi_threaded_result, output_multi_threaded_file)

    # -----------------------------
    # Multi Process (concurrent)
    # ----------------------------

    # Merge sort the array
    start_time = time.time()
    multi_process_result = process_merge_sort(arr, num_processes)
    multi_process_time = time.time() - start_time

    # Feedback of multi process execution time
    logger.debug(
        f"Multi process: merge sort completed in {multi_process_time:.2f} seconds"
    )

    # Save multi process data to a file
    if OUTPUT:
        output_multi_process_file = os.path.join(
            output_folder_path,
            MERGE_MULTI_PROCESS_RESULT_FILE_NAME,
        )

        save_data_to_file(multi_process_result, output_multi_process_file)

    # STDOUT
    print(
        f"{len(arr)} {single_threaded_time} {multi_threaded_time} {multi_process_time}"
    )

    # Returns a tuple:
    # 1. The total number of elements in the array.
    # 2. The execution time for the single-threaded matrix multiplication.
    # 3. The execution time for the multi-threaded matrix multiplication.
    # 4. The execution time for the multi-process matrix multiplication.
    return (
        len(arr),
        single_threaded_time,
        multi_threaded_time,
        multi_process_time,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel Processing in Python: Merge Sort"
    )

    parser.add_argument(
        "array_file_path",
        type=str,
        help="Path to the file containing array",
    )
    parser.add_argument(
        "output_folder_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the output folder for sorted array result (optional)",
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
        array_file_path=args.array_file_path,
        output_folder_path=args.output_folder_path,
        num_threads=args.num_threads,
        num_processes=args.num_processes,
        log_level=args.log_level,
    )