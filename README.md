# Official Implementation for Gaussian Boundary Optimization

This repository contains the source code and visualization results associated with our paper on Gaussian Boundary Optimization (GBO) for weakly supervised video grounding.

![framework](./image/framework.png)

- We propose Gaussian Boundary Optimization (GBO), a novel inference framework for weakly supervised video grounding that formulates segment prediction as a principled optimization problem balancing proposal coverage and segment compactness, addressing the limitations of existing heuristic inference strategies.

- We provide a complete mathematical foundation for GBO, including closed-form solutions to the optimization problem under different penalty weight regimes and rigorous theoretical analysis of optimality conditions. Through comprehensive case analysis, we formally prove the conditions under which the optimal solution yields a non-degenerate segment.

- We demonstrate that GBO is a model-agnostic, training-free inference framework that seamlessly integrates with any Gaussian proposal-based method, including both single-Gaussian and Gaussian mixture representations. Our extensive experiments show that GBO consistently improves localization performance across diverse architectures and datasets, yielding significant gains of up to 11.25\%p. GBO-enhanced models achieve state-of-the-art results without additional training and with negligible inference overhead, making GBO a practical and powerful inference framework.

## Dependencies
- cuda 12.1
- python 3.8
- pytorch 2.0
- nltk
- wandb
- h5py
- fairseq

## Folder Structure

- `gbocnm/`, `gbocpl/`, `gbopps/`  
  Each folder corresponds to a model (**[CNM](https://github.com/minghangz/cnm)**, **[CPL](https://github.com/minghangz/cpl)**, **[PPS](https://github.com/sunoh-kim/pps)**) integrated with GBO.  
  The `train.py` script in each GBO folder is the main entry point for running the corresponding model with GBO inference.  
  You can easily run experiments by executing `eval_gio.sh` in the `script` folder of each GBO folder.  
  **Checkpoints for each model should be downloaded directly from their respective repositories linked above, and should be placed inside the `checkpoint/` folder of each corresponding GBO folder** (e.g., `gbocnm/checkpoint/`, `gbocpl/checkpoint/`, `gbopps/checkpoint/`).

## Example Commands
To run the experiment for the PPS model:
```
cd gbopps/script
bash eval_gio.sh
```

Similarly, you can execute the following for other models:

```
cd gbocpl/script
bash eval_gio.sh
```
```
cd gbocnm/script
bash eval_gio.sh
```

## Visualization

The `fig_data_*.png` files contain performance curves for different models (CNM, CPL, PPS) and datasets (ActivityNet Captions, Charades-STA), evaluated at Rank@1 and Rank@5 under various IoU metrics.


## Acknowledgement
The following repositories were helpful for our implementation.

https://github.com/sunoh-kim/pps

https://github.com/minghangz/cpl

https://github.com/minghangz/cnm


