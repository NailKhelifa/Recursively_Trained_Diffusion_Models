# Recursively Trained Diffusion Models: Limiting Collapse Distribution and Spectral Characterization

Code for the NeurIPS 2026 paper by Nail B. Khelifa, Richard E. Turner and Ramji Venkataramanan.

## Layout

```
2dgmm_hermite/
  hermite2dgmm.py                  Figure 1
2d_gmm/
  main.py                          Figures 2 and 4
cifar_10/
  train_generation0.py             generation-0 score network on CIFAR-10
  recursive_cifar10_training.py    recursive retraining on CIFAR-10
  cifar10_figures.py               Figures 3 and 5
  MULTISEED_..._alphaXX/           training logs of the runs used in the paper
  neurips_figures/                 CIFAR-10 figures
```

The figure PDFs used in the paper are included next to the scripts that produce them.

## Installation

```bash
pip install -r requirements.txt
```

The 2D experiments run on CPU. The CIFAR-10 experiments need a GPU.

## 2D experiments

```bash
cd 2dgmm_hermite && python hermite2dgmm.py
```

Writes `oscillatory_hermite_probe_v2/hermite_pdata_and_theory_limits_row.pdf` (Figure 1).

```bash
cd 2d_gmm && python main.py --out_dir results
```

Writes `14_annealed_truncation.pdf` (Figure 2) and `05_limiting_behaviors.pdf` (Figure 4).
Both figures are computed from the closed-form limit, with Monte Carlo KL estimates averaged
over 5 seeds (mean ± 1 std). The script also trains the learned 2D recursion before plotting;
`--fast` shortens it.

## CIFAR-10 experiments

All commands are run from `cifar_10/`.

**1. Generation 0.** Train a score network on CIFAR-10 only:

```bash
python train_generation0.py --seed 45
```

The best checkpoint (lowest FID) is written to `generation0/seed45/ckpt_best.pt`, which
`recursive_cifar10_training.py` loads by default.

**2. Recursive training.** Set `alpha_real` and `out_dir` in `CFG` at the top of
`recursive_cifar10_training.py`, then run

```bash
python recursive_cifar10_training.py
```

The paper uses `alpha_real` in {0.1, 0.3, 0.5, 0.7, 0.9} with
`out_dir = "MULTISEED_v24_sn96_beta20_ramp20_75_b256_dpm50_t0e3_alphaXX"` (`alpha01`, ...,
`alpha09`), and seed 45. At each generation the training set mixes CIFAR-10 (probability
`alpha_real`) with 50,000 samples from the previous generation, drawn with DPM-Solver++
(50 steps) and stopped at `t0 = 1e-3` (VP time).

Each generation writes to `<out_dir>/ch96/seed45/generation_XXX/` its training log
(`generation_XXX_hist.json`), its weights (`generation_XXX_epoch_300.pt`) and the 50,000
samples used to train the next generation (`synthetic_samples_from_generation_XXX.pt`).
Only the training logs are included in this repository.

**3. Figures.**

```bash
python cifar10_figures.py --runs_dir .              # Figures 3a, 3b and 5
python cifar10_figures.py --figures 5               # Figure 5 only
```

Figures 3a and 3b use the saved samples of generations 0 to 10. Figure 5 samples the
error-free limit directly from CIFAR-10 images and does not need any trained model. In the
plots, code generation `g` is shown as `N = g + 1`. Intermediate results are cached in
`neurips_figures/`.

## Citation

```bibtex
@inproceedings{khelifa2026recursive,
  title     = {Recursively Trained Diffusion Models: Limiting Collapse Distribution and
               Spectral Characterization},
  author    = {Khelifa, Nail B. and Turner, Richard E. and Venkataramanan, Ramji},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
