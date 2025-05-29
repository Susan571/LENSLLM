# <a href="https://arxiv.org/abs/2505.03793" style="color: black !important; text-decoration: none !important;">LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection</a>

**Xinyue Zeng¹**, **Haohui Wang¹**, **Junhong Lin²**, **Jun Wu³**, **Tyler Cody¹**, **Dawei Zhou¹**

¹ Virginia Tech, ² MIT, ³ Michigan State University

<!-- Stylish Buttons -->
<p>
  <img src="Figures/Benchmark.png" alt="DisProtBench" width="90%">
</p>

</div>


LensLLM is a framework for analyzing fine-tuning mechanism and choosing large language model without further training through Neural Tangent Kernel (NTK) theory.

## Features

* NTK matrix tracking during training
* Early stopping optimization using NTK analysis  
* Model selection via zero-shot, subtuning, and LensLLM approaches
* Performance prediction using rectified power law and NTK methods
* Comprehensive analysis tools for model behavior and performance
* Support for various model architectures and training configurations

## Experimental Results

### Fine-tuning Performance Analysis

![Fine-tuning Performance](figures/fine_tuning_performance.png)

The figure above shows the fine-tuning performance across different model sizes and datasets. Our LensLLM approach consistently outperforms baseline methods in terms of both convergence speed and final performance.

### NTK Evolution During Training

![NTK Evolution](figures/ntk_evolution.png)

This visualization demonstrates how the Neural Tangent Kernel evolves during the fine-tuning process, providing insights into model behavior and optimization dynamics.

### Model Selection Comparison

![Model Selection](figures/model_selection.png)

Comparison of different model selection strategies:
- Zero-shot selection
- SubTuning
- ModelSize-based selection
- Our LensLLM approach

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
├── figures/                # Experimental results and visualizations
├── results/               # Saved experimental results
└── README.md             # Project documentation
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LENSLLM.git
cd LENSLLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis notebook:
```bash
jupyter notebook analysis/Analysis.ipynb
```

## Usage Examples

### NTK Tracking During Training

```python
from src.train import train_with_ntk_tracking

# Initialize training with NTK tracking
trainer = train_with_ntk_tracking(
    model=model,
    train_data=train_data,
    ntk_tracking=True,
    save_path='results/ntk_evolution'
)
```

### Model Selection

```python
from src.model_select import lensllm_select

# Select best model using LensLLM
selected_model = lensllm_select(
    models=model_candidates,
    validation_data=val_data,
    selection_criteria='ntk'
)
```

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.