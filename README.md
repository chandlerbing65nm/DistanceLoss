# Naive Distance Loss

This README file provides information about the `distance_loss` Python package, which is built using the provided `setup.py` script.

## Overview

The `distance_loss` package is an extension for PyTorch that provides functionality to compute distance-based losses. This package is built using a combination of C++ and CUDA code for optimal performance on systems with NVIDIA GPUs.

## Requirements

- Python 3.6 or higher
- PyTorch 1.0 or higher
- CUDA Toolkit (if using NVIDIA GPUs)

## Installation

To install the `distance_loss` package, follow these steps:

1. Clone the repository or download the source code.
2. Navigate to the root directory of the project, where the `setup.py` script is located.
3. Run the following command to build and install the package:

```bash
python setup.py build_ext --inplace
```

This command compiles the C++ and CUDA code and creates a Python extension that can be imported as a regular Python package.

## Usage

After installation, you can use the `distance_loss` package in your Python scripts by importing it:
```python
import distance_loss
```

Refer to the package documentation for detailed information on using the provided functionality to compute distance-based losses in your machine learning models.

## Troubleshooting

If you encounter issues during the installation process, make sure you have the correct versions of Python, PyTorch, and the CUDA Toolkit installed on your system. Also, double-check that your environment variables are correctly set up to point to the appropriate locations for your CUDA installation.

If you still encounter issues, please consult the package documentation and seek support from the package maintainers or the community.
