# Script (_5_1_count.py):
# A demonstration of parallel processing for CPU-bound task in Python 3.13.
# This script explores concurrent performance gains and comparisons between threading with and without GIL.
#
# Prerequisites:
# - Python 3.13.1 with experimental free-threading installed in the system environment.
# - Run with GIL: ~/AppData/Local/Programs/Python/Python313/python.exe _5_1_count.py 100000000
# - Run without GIL: ~/AppData/Local/Programs/Python/Python313/python3.13t.exe _5_1_count.py 100000000
#
# Results Summary (counting to 100000000, using 4 threads and 4 processes):
# - With GIL:
#   * Single threaded: completed in 2.89 seconds.
#   * Multi threaded: completed in 3.06 seconds.
#   * Multi process: completed in 0.97 seconds.
# - Without GIL:
#   * Single threaded: completed in 3.33 seconds.
#   * Multi threaded: completed in 0.96 seconds.
#   * Multi process: completed in 1.13 seconds.
#
# Visualized Results (supported by Python 3.12):
# - Use the visualization script to generate a graph of the results.
# - Run: python _5_x_visualization.py ./data_results/visualizations/count.vis
#
# Note:
# - Single-threaded performance is degraded due to additional internal interpreter checks for execution safety.
# - Multi-threaded performance greatly benefits from the absence of GIL.
# - Multi-process performance is similar to the version with GIL but results may vary.
#
# Paper: Parallel Processing – An In-Depth Look Into Python 3.13 (2025)
# Authors: Mantvydas Deltuva and Justinas Teselis

import sys
import time
import logging
import argparse
import threading
import multiprocessing

from _5_x_constants import NUM_THREADS, NUM_PROCESSES, LOG_LEVEL

logger = logging.getLogger(__name__)


# A CPU-bound task: counting to a number
# Single Threaded (consecutive)
def count(n: int, start: int = 0) -> int:
    result = 0
    for _ in range(start, n, 1):
        result += 1
    return result


# Multi Threaded (concurrent)
def threaded_count(n: int, num_threads: int) -> int:
    threads = []
    chunk_size = n // num_threads
    results = [0] * num_threads

    def __worker(index: int, start: int, end: int) -> None:
        results[index] = count(end, start)

    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else n

        thread = threading.Thread(target=__worker, args=(i, start, end))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return sum(results)


# Multi Process (concurrent)
def process_count(n: int, num_processes: int) -> int:
    chunk_size = n // num_processes

    tasks = [
        (
            # end index
            ((i + 1) * chunk_size if i < num_processes - 1 else n),
            # start index
            i * chunk_size,
        )
        for i in range(num_processes)
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(count, tasks)

    return sum(results)


def main(
        count_to: int,
        num_threads: int = NUM_THREADS,
        num_processes: int = NUM_PROCESSES,
        log_level: str = LOG_LEVEL
) -> tuple[int, float, float, float]:
    # Logging configuration (default: debug)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.DEBUG)
    )

    # Feedback of the program status
    logger.debug(
        f"counting to {count_to}, using {num_threads} threads and {num_processes} processes"
    )
    logger.debug(f"The GIL is active: {sys._is_gil_enabled()}")

    # -----------------------------
    # Single Threaded (consecutive)
    # -----------------------------

    # Count to n
    start_time = time.time()
    single_threaded_result = count(count_to)
    single_threaded_time = time.time() - start_time

    # Feedback of single threaded execution time
    logger.debug(
        f"Single threaded: counting to {single_threaded_result} completed in {single_threaded_time:.2f} seconds"
    )

    # ---------------------------
    # Multi Threaded (concurrent)
    # ---------------------------

    # Count to n
    start_time = time.time()
    multi_threaded_result = threaded_count(count_to, num_threads)
    multi_threaded_time = time.time() - start_time

    # Feedback of multi threaded execution time
    logger.debug(
        f"Multi threaded: counting to {multi_threaded_result} completed in {multi_threaded_time:.2f} seconds"
    )

    # --------------------------
    # Multi Process (concurrent)
    # --------------------------

    # Count to n
    start_time = time.time()
    multi_process_result = process_count(count_to, num_processes)
    multi_process_time = time.time() - start_time

    # Feedback of multi process execution time
    logger.debug(
        f"Multi process: counting to {multi_process_result} completed in {multi_process_time:.2f} seconds"
    )

    # STDOUT
    print(
        f"{count_to} {single_threaded_time} {multi_threaded_time} {multi_process_time}"
    )

    # Returns a tuple:
    # 1. The number to count to.
    # 2. The execution time for the single-threaded matrix multiplication.
    # 3. The execution time for the multi-threaded matrix multiplication.
    # 4. The execution time for the multi-process matrix multiplication.
    return (
        count_to,
        single_threaded_time,
        multi_threaded_time,
        multi_process_time,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel Processing in Python: Matrix Multiplication"
    )

    parser.add_argument(
        "count_to",
        type=int,
        help="Number to count to",
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
        count_to=args.count_to,
        num_threads=args.num_threads,
        num_processes=args.num_processes,
        log_level=args.log_level,
    )