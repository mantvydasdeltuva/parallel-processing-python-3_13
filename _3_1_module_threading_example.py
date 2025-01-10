# Script (3.1_module_threading_example.py):
# A demonstration of threading in Python for I/O-bound tasks.
# This script showcases downloading images concurrently from a list of URLs.
#
# Prerequisites:
# - Python 3.12 or higher.
# - Network access for downloading the images from provided URLs.
#
# Paper: Parallel Processing – An In-Depth Look Into Python 3.13 (2025)
# Authors: Mantvydas Deltuva and Justinas Teselis

import threading
import requests
import time
import os

# Create a start event
start_event = threading.Event()


# Universal function for downloading an image in .jpg format from specified URL
def download_image(id, url, output_folder):
    try:
        # Contruct image path based on provided output folder and ID
        image_path = os.path.join(output_folder, f"image_{id}.jpg")

        # Provide initialization feedback
        print(
            f"Thread for image [{id}] is initialized, waiting for start signal..."
        )

        # Wait for start signal
        start_event.wait()

        # Provide start feedback
        print(f"Starting download for image [{id}] from [{url}]")

        # Send a GET request to the URL and stream the response
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Download and save the image in 8KB chunks
        with open(image_path, "wb") as image:
            for chunk in response.iter_content(chunk_size=8192):
                image.write(chunk)

        # Provide completion feedback
        print(f"Completed download for image [{id}], saved to [{image_path}]")
    except Exception as e:
        print(f"Failed to download image [{id}]: {e}")


# List of URLs to download
urls = [
    "https://images.pexels.com/photos/1585325/pexels-photo-1585325.jpeg"
    + "?cs=srgb&dl=pexels-steve-1585325.jpg&fm=jpg",
    "https://images.pexels.com/photos/3246665/pexels-photo-3246665.png"
    + "?cs=srgb&dl=pexels-ekrulila-3246665.jpg&fm=jpg",
    "https://images.pexels.com/photos/1193743/pexels-photo-1193743.jpeg"
    + "?cs=srgb&dl=pexels-paduret-1193743.jpg&fm=jpg",
    "https://images.pexels.com/photos/962312/pexels-photo-962312.jpeg"
    + "?cs=srgb&dl=pexels-minan1398-962312.jpg&fm=jpg",
    "https://images.pexels.com/photos/1762851/pexels-photo-1762851.jpeg"
    + "?cs=srgb&dl=pexels-ann-h-45017-1762851.jpg&fm=jpg",
]

# Output folder for downloaded images
output = os.path.expanduser("~\\Downloads")
os.makedirs(output, exist_ok=True)

# Create threads for downloading images
threads = [
    threading.Thread(target=download_image, args=(i, url, output))
    for i, url in enumerate(urls)
]

# Initialize all threads
for thread in threads:
    thread.start()

# Signal all threads to start
time.sleep(3)
print("Releasing threads to start downloads...")
start_event.set()

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Provide finish feedback
print("All downloads completed.")
