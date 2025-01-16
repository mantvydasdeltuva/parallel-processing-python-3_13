import os
from pathlib import Path


# ------------------------
# Global Environment Paths
# ------------------------
PYTHON_313_PATH = os.path.expanduser(
    "~/AppData/Local/Programs/Python/Python313/python.exe"  # TODO Make compatible with MacOS and Linux
)
PYTHON_313t_PATH = os.path.expanduser(
    "~/AppData/Local/Programs/Python/Python313/python3.13t.exe"  # TODO Make compatible with MacOS and Linux
)

# -----------------------
# Local Environment Paths
# -----------------------
BASE_PATH = Path(__file__).resolve().parent
IO_FOLDER_PATH = os.path.join(BASE_PATH, "data_results")
VISUALIZATIONS_FOLDER_PATH = os.path.join(IO_FOLDER_PATH, "visualizations")

# Count
COUNT_SCRIPT_PATH = os.path.join(BASE_PATH, "_5_1_count.py")
COUNT_IO_FOLDER_PATH = os.path.join(IO_FOLDER_PATH, "_5_1")

# Download Images
DOWNLOAD_SCRIPT_PATH = os.path.join(BASE_PATH, "_5_2_download_images.py")
DOWNLOAD_IO_FOLDER_PATH = os.path.join(IO_FOLDER_PATH, "_5_2")

# Merge Sort
MERGE_SCRIPT_PATH = os.path.join(BASE_PATH, "_5_3_merge_sort.py")
MERGE_IO_FOLDER_PATH = os.path.join(IO_FOLDER_PATH, "_5_3")
MERGE_FILE_PATH = os.path.join(MERGE_IO_FOLDER_PATH, "merge_sort_sample_1m.txt")

# Matrix Multiplication
MATRIX_SCRIPT_PATH = os.path.join(BASE_PATH, "_5_4_matrix_multiplication.py")
MATRIX_IO_FOLDER_PATH = os.path.join(IO_FOLDER_PATH, "_5_4")
MATRIX_A_FILE_PATH = os.path.join(
    MATRIX_IO_FOLDER_PATH, "matrix_multiplication_sample_a.txt"
)
MATRIX_B_FILE_PATH = os.path.join(
    MATRIX_IO_FOLDER_PATH, "matrix_multiplication_sample_b.txt"
)

# ---------
# Constants
# ---------
NUM_THREADS = 4
NUM_PROCESSES = 4
LOG_LEVEL = "debug"

# Download Images
DOWNLOAD_API_BASE_URL = "https://civitai.com/api/v1/images?page=1&limit={}"
DOWNLOAD_SINGLE_THREADED_RESULT_FOLDER_NAME = (
    "download_images_single_threaded_results"
)
DOWNLOAD_MULTI_THREADED_RESULT_FOLDER_NAME = (
    "download_images_multi_threaded_results"
)
DOWNLOAD_MULTI_PROCESS_RESULT_FOLDER_NAME = (
    "download_images_multi_process_results"
)

# Merge Sort
MERGE_SINGLE_THREADED_RESULT_FILE_NAME = "merge_sort_single_threaded_result.txt"
MERGE_MULTI_THREADED_RESULT_FILE_NAME = "merge_sort_multi_threaded_result.txt"
MERGE_MULTI_PROCESS_RESULT_FILE_NAME = "merge_sort_multi_process_result.txt"

# Matrix Multiplication
MATRIX_SINGLE_THREADED_RESULT_FILE_NAME = "matrix_multiplication_single_threaded_result.txt"
MATRIX_MULTI_THREADED_RESULT_FILE_NAME = "matrix_multiplication_multi_threaded_result.txt"
MATRIX_MULTI_PROCESS_RESULT_FILE_NAME = "matrix_multiplication_multi_process_result.txt"

# Visualization
VISUALIZATION_FILE_EXTENSION = ".vis"
SINGLE_THREADED_GIL_KEY = "single_threaded_gil"
SINGLE_THREADED_KEY = "single_threaded"
SINGLE_THREAD_PERFORMANCE_KEY = "single_thread_performance"
MULTI_THREADED_GIL_KEY = "multi_threaded_gil"
MULTI_THREADED_KEY = "multi_threaded"
MULTI_THREAD_PERFORMANCE_KEY = "multi_thread_performance"
MULTI_PROCESS_GIL_KEY = "multi_process_gil"
MULTI_PROCESS_KEY = "multi_process"
MULTI_PROCESS_PERFORMANCE_KEY = "multi_process_performance"
KEYS = [
    SINGLE_THREADED_GIL_KEY,
    MULTI_THREADED_GIL_KEY,
    MULTI_PROCESS_GIL_KEY,
    SINGLE_THREADED_KEY,
    MULTI_THREADED_KEY,
    MULTI_PROCESS_KEY,
    SINGLE_THREAD_PERFORMANCE_KEY,
    MULTI_THREAD_PERFORMANCE_KEY,
    MULTI_PROCESS_PERFORMANCE_KEY,
]
LABELS = {
    SINGLE_THREADED_GIL_KEY: "Single Threaded With GIL",
    MULTI_THREADED_GIL_KEY: "Multi Threaded With GIL",
    MULTI_PROCESS_GIL_KEY: "Multi Process With GIL",
    SINGLE_THREADED_KEY: "Single Threaded Without GIL",
    MULTI_THREADED_KEY: "Multi Threaded Without GIL",
    MULTI_PROCESS_KEY: "Multi Process Without GIL",
    SINGLE_THREAD_PERFORMANCE_KEY: "Single Threaded",
    MULTI_THREAD_PERFORMANCE_KEY: "Multi Threaded",
    MULTI_PROCESS_PERFORMANCE_KEY: "Multi Process",
}
COLORS = {
    SINGLE_THREADED_GIL_KEY: "#FF8566",
    MULTI_THREADED_GIL_KEY: "#95BFE8",
    MULTI_PROCESS_GIL_KEY: "#BBE8AB",
    SINGLE_THREADED_KEY: "#FF3300",
    MULTI_THREADED_KEY: "#4E95D9",
    MULTI_PROCESS_KEY: "#8ED873",
    SINGLE_THREAD_PERFORMANCE_KEY: "#FF3300",
    MULTI_THREAD_PERFORMANCE_KEY: "#4E95D9",
    MULTI_PROCESS_PERFORMANCE_KEY: "#8ED873",
}

# ------
# Errors
# ------
if not PYTHON_313_PATH:
    raise FileNotFoundError("Python 3.13 executable not found.")
if not PYTHON_313t_PATH:
    raise FileNotFoundError("Python 3.13t executable not found.")
