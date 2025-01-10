# Script (_5_2_download_images.py):
# A demonstration of parallel processing for I/O-bound task in Python 3.13.
# This script explores concurrent performance gains and comparisons between threading with and without GIL.
#
# Prerequisites:
# - Python 3.13.1 with experimental free-threading installed in the system environment.
# - Network access for downloading the images from provided URLs.
# - Run with GIL: ~/AppData/Local/Programs/Python/Python313/python.exe _5_2_download_images.py 200
# - Run without GIL: ~/AppData/Local/Programs/Python/Python313/python3.13t.exe _5_2_download_images.py 200
#
# Results Summary (200 urls, using 4 threads and 4 processes):
# - With GIL:
#   * Single threaded: completed in 20.74 seconds.
#   * Multi threaded: completed in 9.68 seconds.
#   * Multi process: completed in 11.84 seconds.
# - Without GIL:
#   * Single threaded: completed in 19.94 seconds.
#   * Multi threaded: completed in 8.19 seconds.
#   * Multi process: completed in 8.97 seconds.
#
# Visualized Results (supported by Python 3.12):
# - Use the visualization script to generate a graph of the results.
# - Run: python _5_x_visualization.py ./data_results/visualizations/download_images.vis
#
# Note:
# - Single-threaded performance is slightly better without GIL due to reduced interpreter overhead.
# - Multi-threaded performance benefits are minor from the absence of GIL.
# - Multi-process performance fluctuates without GIL.
#
# Paper: Parallel Processing – An In-Depth Look Into Python 3.13 (2025)
# Authors: Mantvydas Deltuva and Justinas Teselis

import os
import sys
import time
import logging
import argparse
import requests
import threading
import multiprocessing

from _5_x_constants import (
    NUM_THREADS,
    NUM_PROCESSES,
    LOG_LEVEL,
    DOWNLOAD_IO_FOLDER_PATH,
    DOWNLOAD_SINGLE_THREADED_RESULT_FOLDER_NAME,
    DOWNLOAD_MULTI_THREADED_RESULT_FOLDER_NAME,
    DOWNLOAD_MULTI_PROCESS_RESULT_FOLDER_NAME,
    DOWNLOAD_API_BASE_URL,
)

logger = logging.getLogger(__name__)


# An I/O-bound task: downloading image
def __download_image(url: str, output_path: str) -> int:
    size = 0
    try:
        response = requests.get(url, timeout=2, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                size += len(chunk)
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
    return size


# Single Threaded (consecutive)
def download_images(urls: list[str], output_folder_path: str) -> int:
    size = 0
    output_folder_path = os.path.join(
        output_folder_path, DOWNLOAD_SINGLE_THREADED_RESULT_FOLDER_NAME
    )
    os.makedirs(output_folder_path, exist_ok=True)
    for url in urls:
        output_path = os.path.join(output_folder_path, url.split("/")[-1])
        size += __download_image(url, output_path)
    return size


# Multi Threaded (concurrent)
def threaded_download_images(
    urls: list[str], output_folder_path: str, num_threads: int
) -> None:
    threads = []
    chunk_size = len(urls) // num_threads

    output_folder_path = os.path.join(
        output_folder_path, DOWNLOAD_MULTI_THREADED_RESULT_FOLDER_NAME
    )
    os.makedirs(output_folder_path, exist_ok=True)

    def __worker(start: int, end: int):
        for url in urls[start:end]:
            output_path = os.path.join(output_folder_path, url.split("/")[-1])
            __download_image(url, output_path)

    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else len(urls)

        thread = threading.Thread(target=__worker, args=(start, end))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


# Multi Process (concurrent)
def process_download_images(
    urls: list[str], output_folder_path: str, num_processes: int
) -> None:
    output_folder_path = os.path.join(
        output_folder_path, DOWNLOAD_MULTI_PROCESS_RESULT_FOLDER_NAME
    )
    os.makedirs(output_folder_path, exist_ok=True)

    tasks = [
        (
            # url string
            url,
            # output path string
            os.path.join(
                output_folder_path,
                url.split("/")[-1],
            ),
        )
        for url in urls
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(__download_image, tasks)


def __fetch_urls(urls_count: int) -> list[str]:
    if urls_count < 1 or urls_count > 200:
        raise ValueError("The number of URLs must be between 1 and 200.")

    api_url = DOWNLOAD_API_BASE_URL.format(urls_count)
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        return [item["url"] for item in data["items"]]
    except Exception as e:
        logger.error(f"Failed to retrieve urls. Error: {e}")


def main(
    urls_count: int,
    output_folder_path: str = DOWNLOAD_IO_FOLDER_PATH,
    num_threads: int = NUM_THREADS,
    num_processes: int = NUM_PROCESSES,
    log_level: str = LOG_LEVEL,
) -> tuple[int, float, float, float]:
    # Logging configuration (default: debug)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.DEBUG)
    )
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # Variables
    urls = __fetch_urls(urls_count)

    # Feedback of the program status
    logger.debug(
        f"{urls_count} urls, using {NUM_THREADS} threads and {NUM_PROCESSES} processes"
    )
    logger.debug(f"The GIL is active: {sys._is_gil_enabled()}")

    # -----------------------------
    # Single Threaded (consecutive)
    # -----------------------------

    # Download images
    start_time = time.time()
    size = download_images(urls, output_folder_path)
    single_threaded_time = time.time() - start_time

    # Feedback of single threaded execution time
    logger.debug(
        f"Single threaded: download images completed in {single_threaded_time:.2f} seconds"
    )

    # ---------------------------
    # Multi Threaded (concurrent)
    # ---------------------------

    # Download images
    start_time = time.time()
    threaded_download_images(urls, output_folder_path, num_threads)
    multi_threaded_time = time.time() - start_time

    # Feedback of multi threaded execution time
    logger.debug(
        f"Multi threaded: download images completed in {multi_threaded_time:.2f} seconds"
    )

    # --------------------------
    # Multi Process (concurrent)
    # --------------------------

    # Download images
    start_time = time.time()
    process_download_images(urls, output_folder_path, num_processes)
    multi_process_time = time.time() - start_time

    # Feedback of multi process execution time
    logger.debug(
        f"Multi process: download images completed in {multi_process_time:.2f} seconds"
    )

    # STDOUT
    print(
        f"{size} {single_threaded_time} {multi_threaded_time} {multi_process_time}"
    )

    # Returns a tuple:
    # 1. The total size of bytes that were downloaded.
    # 2. The execution time for the single-threaded matrix multiplication.
    # 3. The execution time for the multi-threaded matrix multiplication.
    # 4. The execution time for the multi-process matrix multiplication.
    return (
        size,
        single_threaded_time,
        multi_threaded_time,
        multi_process_time,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel Processing in Python: Download Images"
    )

    parser.add_argument(
        "urls_count",
        type=int,
        choices=range(1, 201),
        help="Number of images to download",
    )
    parser.add_argument(
        "-o",
        "--output_folder_path",
        type=str,
        default=DOWNLOAD_IO_FOLDER_PATH,
        help="Path to the output folder for downloaded images (default: ./data_results/_5_2)",
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
        urls_count=args.urls_count,
        output_folder_path=args.output_folder_path,
        num_threads=args.num_threads,
        num_processes=args.num_processes,
        log_level=args.log_level,
    )
