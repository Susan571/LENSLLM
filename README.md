# LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection

LensLLM is a framework for analyzing fine-tuning mechanism and choosing large language model without further training through Neural Tangent Kernel (NTK) theory.

## Features

* NTK matrix tracking during training
* Early stopping optimization using NTK analysis  
* Model selection via zero-shot, subtuning, and LensLLM approaches
* Performance prediction using rectified power law and NTK methods
* Comprehensive analysis tools for model behavior and performance
* Support for various model architectures and training configurations

## Project Structure

```
.
├── analysis/                 # Analysis notebooks and scripts
│   ├── Analysis.ipynb       # Main analysis notebook with figures
│   └── analysis_utils.py    # Analysis utility functions
├── src/                     # Source code
│   ├── train.py            # Training loop with NTK tracking
│   ├── model_select.py     # Model selection strategies
│   ├── fit_law.py          # Power law and LensLLM fitting
│   ├── dataset.py          # Data handling
│   └── utils/              # Utility modules
│       ├── func_utils.py   # Model utilities
│       ├── custom_utils.py # Training components  
│       ├── const_utils.py  # Constants
│       └── env_utils.py    # Environment setup
└── README.md               # Project documentation
```

## Getting Started

1. Clone the repository
2. Install dependencies
3. Run the analysis notebook to explore the framework's capabilities

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zeng2025lensllm,
  title={LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection},
  author={Zeng, Xinyue and Wang, Haohui and Lin, Junhong and Wu, Jun and Cody, Tyler and Zhou, Dawei},
  journal={arXiv preprint arXiv:2505.03793},
  year={2025}
}
```