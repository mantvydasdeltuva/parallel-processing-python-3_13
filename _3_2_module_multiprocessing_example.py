# Script (3.2_module_multiprocessing_example.py):
# A demonstration of multiprocessing in Python for CPU-bound.
# This script showcases parallel computation of squares using a process pool, simulating heavy computation with random delays.
#
# Prerequisites:
# - Python 3.12 or higher.
#
# Paper: Parallel Processing – An In-Depth Look Into Python 3.13 (2025)
# Authors: Mantvydas Deltuva and Justinas Teselis

from multiprocessing import Pool, current_process
import random
import time


# Universal function to compute the square of a number
def compute_square(number):
    # Provide start feedback
    process_name = current_process().name
    sleep_time = random.uniform(0.5, 2.0)
    print(
        f"Process {process_name} is calculating square "
        + f"for {number} with sleep time {sleep_time}"
    )

    # Simulate a heavy computation with a random sleep time
    time.sleep(sleep_time)

    # Return the square of the number
    return number * number


if __name__ == "__main__":
    # Numbers to calculate the square of
    numbers = [1, 512, 4, 128, 16, 32, 64, 8, 256, 2]

    # Create a pool of processes to calculate the squares
    with Pool(processes=4) as pool:
        results = pool.map(compute_square, numbers)

    # Provide finish feedback
    print(f"Squares: {results}")
