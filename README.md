# <a href="https://arxiv.org/abs/2505.03793" style="color: black !important;">LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection[ICML 2025]🎉🎉🎉</a>

**Xinyue Zeng¹**, **Haohui Wang¹**, **Junhong Lin²**, **Jun Wu³**, **Tyler Cody¹**, **Dawei Zhou¹**

¹ Virginia Tech, ² MIT, ³ Michigan State University

---

## 📌 Abstract
The proliferation of open-sourced Large Language Models (LLMs) and diverse downstream tasks necessitates efficient model selection, given the impracticality of fine-tuning all candidates due to computational constraints. Despite the recent advances in LLM selection, a fundamental research question largely remains nascent: *how can we model the dynamic behaviors of LLMs during fine-tuning, thereby enhancing our understanding of their generalization performance across diverse downstream tasks?*

In this work, we propose a novel theoretical framework that provides a proper lens to assess the generalization capabilities of LLMs, thereby enabling accurate and efficient LLM selection for downstream applications. Our key contributions include:

(1) Deriving a **PAC-Bayesian Generalization Bound** that unveils the fine-tuning dynamics of LLMs

(2) Introducing **LENSLLM**, a Neural Tangent Kernel (NTK)-based Rectified Scaling Model that enables accurate performance predictions across diverse tasks while maintaining computational efficiency

(3) Demonstrating through extensive empirical results on three large-scale benchmarks that our model achieves **up to 91.1% accuracy** and reduces **up to 88.5% computational cost** in LLM selection, outperforming five state-of-the-art methods

## 🔍 Understanding Phase Transitions in LLM Fine-tuning

Our analysis uncovers two distinct phases in the fine-tuning dynamics of Large Language Models (LLMs), each with unique implications for model selection. These phases—marked by shifts in sensitivity, performance scaling, and Neural Tangent Kernel (NTK) evolution—play a critical role in understanding and predicting model behavior.

<div align="center">
<img src="Figure/model_comparison.png" width="90%">
<p><em>Figure: Fine-tuning test loss (L) as a function of training sample size (D). The curve highlights an initial pre-power phase at smaller D, followed by a power phase exhibiting a clear linear trend in log-log scale.</em></p>
</div>

<div align="center">

| Phase | Description | Key Characteristics |
|:----------|:----------------|:------------------------|
| **Pre-power Phase** | Early stage of fine-tuning with rapid performance shifts | • High sensitivity to parameter updates<br>• Non-linear, often dramatic improvements<br>• Significant NTK matrix evolution<br>• Dynamic and task-specific behavior |
| **Power Phase** | Later stage where performance scales predictably | • Stable and predictable performance improvements<br>• Power-law relationship between data size and loss<br>• Stabilized NTK structure<br>• Consistent behavior across tasks |
</div>

### Why This Matters for Model Selection

Understanding and identifying these phases enables:
1. **Phase-aware performance prediction**: Accounting for the current phase improves extrapolations.
2. **Adaptive model selection**: Different models may reach the power phase at different data thresholds.
3. **Improved efficiency**: Fine-tuning strategies can be tailored to the phase, reducing redundant computation.
4. **Theoretical interpretability**: NTK stability serves as a signal for transition detection.

Our **LENSLLM** framework explicitly models both phases, offering accurate performance estimation and phase-adaptive model selection across a range of fine-tuning scenarios.

## 📊 Theoretical Foundation: PAC-Bayesian Generalization Bound

Our theoretical framework establishes a fundamental link between fine-tuning dynamics and model generalization through a novel PAC-Bayesian bound. Specifically, for any $\epsilon > 0$, with probability at least 0.99, and under our stated assumptions, the generalization error satisfies:

$$
L(f_{\hat{w}}) \leq (1 + \epsilon)\hat{L}(f_{\hat{w}}) + C n^{-\beta} + O(n^{-\frac{3}{4}})
$$

Here, $C$ and $\beta$ are constants that depend on the model architecture and downstream task.

This bound serves as the theoretical basis for characterizing the **two distinct phases** observed during LLM fine-tuning:

<div align="center">

| Phase | Error Scaling | Key Characteristics | Model Behavior | Data Requirements |
|:----------|:-----------------|:------------------------|:-------------------|:---------------------|
| **Pre-power Phase** | $O(n^{-\frac{3}{4}})$ | • Large Hessian values<br>• High sensitivity to parameter changes<br>• Non-linear performance gains | • Gradual improvements<br>• Task-dependent variability<br>• Unstable convergence patterns | • Requires more data<br>• Sensitive to hyperparameters<br>• Slower convergence |
| **Power Phase** | $Cn^{-\beta}$ | • Lower Hessian values<br>• Stabilized gradients<br>• Predictable scaling behavior | • Consistent improvements<br>• Task-agnostic trends<br>• Scalable performance across data regimes | • Greater data efficiency<br>• Robust to learning rate choices<br>• Fewer training epochs needed |
</div>

### 🔄 Phase Transition Dynamics and the LENSLLM Framework

Our theoretical analysis reveals a critical phase transition in LLM fine-tuning, characterized by a shift in the dominant term of the generalization bound:

- **From**: $O(n^{-\frac{3}{4}})$ — governing the early, unstable fine-tuning regime  
- **To**: $Cn^{-\beta}$ — capturing the later, stable scaling behavior  

This transition reflects a **reduction in Hessian magnitude** and **parameter sensitivity**, signaling a progression from chaotic to stable learning dynamics.

To model this transition, we introduce **LENSLLM**, a Hessian-aware rectified scaling framework that captures the evolving dynamics of fine-tuning and enables accurate generalization prediction and efficient model selection across diverse tasks.

Motivated by the alignment between theory and empirical observations, we formulate LENSLLM as:

$$
L(D) = \frac{B}{F(\mathbf{\Theta}, t) + D^{\beta}} + E
$$

where:
- $F(\mathbf{\Theta}, t)$ is a task- and architecture-adapted NTK-based test loss function for transformers,  
- $D$ is the number of training samples,  
- $\beta$ represents task-specific learning difficulty,  
- $B$ controls the initial test loss level, and  
- $E$ denotes the asymptotic optimal loss achievable with unlimited data.

This formulation allows LENSLLM to generalize well across data scales and model architectures, offering practical guidance for phase-aware model selection.

## 📈 Empirical Results

We validate our theoretical framework through comprehensive experiments across a diverse set of language models and three major datasets. Our results demonstrate both the **predictive accuracy** and **efficiency gains** enabled by LENSLLM.

<div align="center">
<img src="Figure/combined_plots.png" width="90%">
<p><em>Figure: Performance comparison showing the superior effectiveness of LENSLLM (blue squares) across OPT-1.3B, GPT-2, and T5-Base on the FLAN, Wikitext, and Gigaword datasets. LENSLLM consistently achieves lower RMSE values compared to the Rectified Scaling Law (red triangles), with narrower error bands indicating more stable performance.</em></p>
</div>

<div align="center">

| Model | Wikitext | FLAN | Gigaword |
|:----------|:------------:|:--------:|:------------:|
| | Ours | Rect | Ours | Rect | Ours | Rect |
| OPT-350M | **0.20** | 1.10 | **0.32** | 1.50 | **0.26** | 0.98 |
| OPT-1.3B | **0.32** | 1.14 | **0.32** | 1.20 | **0.28** | 0.99 |
| OPT-6.7B | **0.26** | 1.32 | **0.26** | 1.31 | **0.26** | 1.46 |
| T5-Small | **0.35** | 1.01 | **0.28** | 1.30 | **0.30** | 1.27 |
| T5-Base | **0.32** | 1.30 | **0.26** | 1.26 | **0.30** | 0.94 |
| Cerebras-256M | **0.24** | 1.27 | **0.22** | 1.10 | **0.33** | 1.30 |
| Cerebras-1.3B | **0.26** | 1.18 | **0.32** | 1.00 | **0.28** | 1.00 |
| mT5-Base | **0.26** | 1.17 | **0.32** | 1.22 | **0.17** | 1.07 |
| mT5-Large | **0.28** | 1.44 | **0.32** | 1.07 | **0.28** | 1.10 |
| BART-Base | **0.30** | 1.27 | **0.30** | 0.96 | **0.26** | 0.99 |
| BART-Large | **0.17** | 1.31 | **0.28** | 0.87 | **0.36** | 1.14 |
| GPT-2 | **0.30** | 1.30 | **0.30** | 1.23 | **0.26** | 1.33 |
| LaMini-124M | **0.28** | 1.01 | **0.35** | 1.00 | **0.30** | 1.15 |
| LaMini-774M | **0.32** | 1.14 | **0.28** | 1.13 | **0.28** | 1.19 |
</div>

*Table: Root Mean Squared Error (RMSE) comparison between predicted and actual test losses ($\times 10^{-1}$) of our model and Rectified Scaling Law. Lower values indicate better prediction accuracy.*

<div align="center">
<img src="Figure/model_selection_comparison_JL.png" width="80%">
<p><em>Figure: Comparison of model selection approaches. LENSLLM consistently identifies optimal models with higher accuracy and robustness compared to baseline methods.</em></p>
</div>

<div align="center">
<img src="Figure/Revised_gigaword_plot.png" width="80%">
<p><em>Figure: Resource efficiency on the Gigaword dataset. LENSLLM achieves comparable or better performance with substantially reduced computational cost.</em></p>
</div>

---

### 📌 Key Takeaways

- **Superior Prediction Accuracy**: LENSLLM consistently outperforms the Rectified Scaling Law across all evaluated models and datasets, achieving up to **5× lower RMSE**.
- **Robust Across Scales**: The performance gap holds across both small (e.g., OPT-350M, T5-Small) and large (e.g., OPT-6.7B, mT5-Large) models.
- **Practical Efficiency**: The reduced prediction error enables more informed and computationally efficient model selection, particularly valuable when fine-tuning resources are limited.

Together, these results provide strong empirical support for the practical utility of LENSLLM in **scaling law prediction**, **test-time selection**, and **resource-aware fine-tuning**.

## 📁 Project Structure

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

### 🚀 Getting Started

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

#### NTK Tracking During Training

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

#### Model Selection

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
