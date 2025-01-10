# Script (3.3_module_concurrent.futures_example.py):
# A demonstration of parallel processing using `concurrent.futures` in Python.
# This script utilizes `ProcessPoolExecutor` for efficient computation of factorials.
#
# Prerequisites:
# - Python 3.12 or higher.
#
# Paper: Parallel Processing – An In-Depth Look Into Python 3.13 (2025)
# Authors: Mantvydas Deltuva and Justinas Teselis


from concurrent.futures import ProcessPoolExecutor
from multiprocessing import current_process
import random
import time
import math


# Universal function to compute factorial of a number
def compute_factorial(number):
    # Provide start feedback
    process_name = current_process().name
    sleep_time = random.uniform(0.5, 2.0)
    print(
        f"Process {process_name} is calculating factorial "
        + f"for {number} with sleep time {sleep_time}"
    )

    # Simulate a heavy computation with a random sleep time
    time.sleep(sleep_time)

    # Return the factorial of the number
    return math.factorial(number)


if __name__ == "__main__":
    # Numbers to calculate the factorial of
    numbers = [1, 16, 4, 5, 8, 6, 2]

    # Create ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(compute_factorial, numbers))

    # Provide finish feedback
    print(f"Results: {results}")
