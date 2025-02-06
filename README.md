# LensLLM

LensLLM is a framework for analyzing and optimizing large language model training through Neural Tangent Kernel (NTK) theory.

## Features

* NTK matrix tracking during training
* Early stopping optimization using NTK analysis  
* Model selection via zero-shot, subtuning, and LensLLM approaches
* Performance prediction using rectified power law and NTK methods

## Project Structure

```
.
├── Analysis.ipynb     # Analysis notebook with figures
├── train.py           # Training loop with NTK tracking
├── model_select.py    # Model selection strategies
├── fit_law.py         # Power law and LensLLM fitting
├── dataset.py         # Data handling
└── utils/
    ├── func_utils.py  # Model utilities
    ├── custom_utils.py # Training components  
    ├── const_utils.py # Constants
    └── env_utils.py   # Environment setup
```