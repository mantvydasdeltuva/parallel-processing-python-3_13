<div align="center">
  <img src="assets/banner-dark.png#gh-dark-mode-only" alt="Banner" style="width: 600px; height: auto;">
  <img src="assets/banner-light.png#gh-light-mode-only" alt="Banner" style="width: 600px; height: auto;">
</div>

---

### Introduction

This repository contains the source code and documentation for a project exploring the advancements in parallel processing introduced in **Python 3.13**. With the experimental **removal of the Global Interpreter Lock (GIL)**, this version unlocks new possibilities for achieving true parallelism in multi-threaded applications. This document delves into the concepts, techniques, and performance implications of parallel processing in Python 3.13, comparing traditional GIL-based execution with the new GIL-free mode.

---

### Background

The document provides a background on the **evolution of parallel processing in Python**, covering key modules like `threading`, `multiprocessing`, and `concurrent.futures`. It explains the concepts of **threads** and **processes**, highlighting their strengths and limitations, particularly in the context of the GIL.

---

### Free-Threaded Mode

**Python 3.13 introduces an experimental free-threaded mode**, which disables the GIL, allowing multiple threads to execute Python bytecode simultaneously. This feature significantly **improves Python's ability to utilize multi-core processors for parallel execution**, leading to potential performance gains in multi- threaded applications. The document discusses the benefits and challenges associated with this new mode, including its experimental nature and potential compatibility concerns.

---

### Performance Comparison

The document presents a detailed **performance comparison between GIL-enabled and GIL-free execution models** across various scenarios, including:

- **Count:** A simple counting algorithm to demonstrate the impact of the GIL on CPU-bound tasks.
- **Download Images:** An I/O-bound task showcasing the behavior of threading and multiprocessing in both GIL-enabled and GIL-free environments.
- **Merge Sort:** A classic sorting algorithm to evaluate the performance of parallel implementations.
- **Matrix Multiplication:** A computationally intensive task highlighting the potential performance benefits of removing the GIL.

The results of these benchmarks are presented with **graphical visualizations**, providing insights into the **performance trade-offs** between different approaches and the influence of the GIL on parallel execution.

---

### Conclusion

The document concludes by **emphasizing the significance of removing the GIL** in Python 3.13 and its impact on the future of the language. It acknowledges the **challenges and ongoing development** required to fully leverage this new feature while expressing optimism about the potential for **faster and more scalable Python applications**.

---

### Repository Contents

- `README.md`: This file, providing an overview of the project.
- `Parallel-Processing-in-Python-3_13.pdf`: The complete document exploring parallel processing in Python 3.13.
- `Parallel-Processing-in-Python-3_13-Presentation.pdf`: Presentation of this project.
- `src`: Folder that contains the Python scripts used for benchmarking and demonstrating parallel processing techniques.

---

### License

This project is licensed under the MIT License. See the LICENSE file for more details.
