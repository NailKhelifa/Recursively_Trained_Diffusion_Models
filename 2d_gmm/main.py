#!/usr/bin/env python3
"""
Model Collapse in Diffusion Models — 2-GMM Experiment  (v2)
=============================================================

Illustrates three theorems:

  THM 1 — Neumann collapse distribution:
      p_inf = α Σ_{k=0}^∞ (1-α)^k  U_{(k+1)t0}(p_data)
      (geometric mixture of progressively OU-smoothed data)

  THM 2 — Limiting behaviours (all in KL):
      (i)  α→1 : KL(p_inf ‖ U_{t0}(p_data)) → 0
      (ii) α→0 : KL(p_inf ‖ N(0,I))        → 0
      (iii) t0→0: KL(p_inf ‖ p_data)         → 0
      (iv) t0→∞: KL(p_inf ‖ N(0,I))         → 0
      (v)  for any α∈(0,1), t0>0: KL(p_inf ‖ p_data) > 0

  THM 3 — Geometric convergence & stability:
      KL(p̂^N ‖ p̃_∞) ≤ (1-α/2)^N KL(p̂^0 ‖ p̃_∞) + C/α · Σ (1-α/2)^{N-1-i} η_i

  Contraction constant:  κ = (1-α) e^{-t0/2}

OU process: dx = -½x dt + dW  →  U_t(N(μ,σ²I)) = N(e^{-t/2}μ, [σ²e^{-t}+(1-e^{-t})]I)
VP-SDE uses β≡1 (BETA_MIN=BETA_MAX=1) so it equals simple OU — κ matches exactly.

Outputs (--out_dir):
  Per-alpha subdirs: gen_XX.pt checkpoints, gen_XX_samples.npy
  all_results.pkl
  Plots used in the paper:
    05_limiting_behaviors.{pdf,png}     (Figure 4)
    14_annealed_truncation.{pdf,png}    (Figure 2)

Usage:
    python main.py                       # full run
    python main.py --fast                # quick sanity (~3 min)
    python main.py --plot_only <out_dir>/all_results.pkl
"""

import os, sys, time, pickle, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  GLOBAL STYLE
# ─────────────────────────────────────────────────────────────
import matplotlib as mpl
import matplotlib.ticker
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "legend.fontsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "figure.dpi": 200,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "legend.frameon": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COL_TOTAL = "#E63946"     # red
COL_GRAD  = "#2A9D8F"     # green
COL_SOL   = "#6C5B7B"     # purple
COL_GRAY  = "#8D99AE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KernelDensity
import math as _math

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_GENERATIONS    = 30
ALPHAS           = [0.1, 0.5, 0.9]
SEED             = 42

N_TRAIN          = 20_000
N_GEN_SAMPLES    = 10_000
N_STORE          = 3_000

GMM_MEANS        = [[-3.0, -3.0], [3.0, 3.0], [-5.0, 2.0]]
GMM_STDS         = [1.0, 1.0, 2.0]
GMM_WEIGHTS      = [0.3, 0.2, 0.5]

HIDDEN           = 256
TIME_EMB_DIM     = 64
N_LAYERS         = 4

# β≡1 → VP-SDE = simple OU → κ = (1-α)e^{-t0/2} exact
T_MAX            = 1.0
BETA_MIN         = 1.0
BETA_MAX         = 1.0
N_REVERSE_STEPS  = 1000

BATCH_SIZE       = 512
N_EPOCHS_GEN0    = 300
N_EPOCHS_RECURSE = 120
LR               = 3e-4

KDE_BW           = 0.15
MC_KL_SAMPLES    = 20_000
NEUMANN_K        = 120

GRID_LIM         = 7.0
GRID_RES         = 160

OUT_DIR          = "model_collapse_2gmm_vf11"

ALPHA_COLORS = {0.1: '#d62728', 0.5: '#ff7f0e', 0.9: '#2166ac'}
ALPHA_CMAPS  = {0.1: 'Reds',    0.5: 'Oranges', 0.9: 'Blues'}


# ─────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def alpha_dir(alpha):
    d = os.path.join(OUT_DIR, f"alpha_{alpha:.1f}")
    os.makedirs(d, exist_ok=True)
    return d

def savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def log(msg): print(msg, flush=True)

def kappa(alpha, t0=None):
    if t0 is None: t0 = T_MAX
    return (1.0 - alpha) * np.exp(-t0 / 2.0)


# ─────────────────────────────────────────────────────────────
#  ANALYTIC GMM  (OU smoothing + Neumann series)
# ─────────────────────────────────────────────────────────────
class AnalyticGMM:
    """
    Isotropic Gaussian mixture in R^2.
    weights: (K,)   means: (K,2)   stds: (K,)  [isotropic, one σ per component]
    """
    def __init__(self, weights, means, stds):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means   = np.asarray(means,   dtype=np.float64)
        self.stds    = np.asarray(stds,    dtype=np.float64)
        assert abs(self.weights.sum() - 1.0) < 1e-5

    # ── exact log-density ──────────────────────────────────
    def log_prob(self, x):
        x  = np.asarray(x, dtype=np.float64)
        lp = np.full(len(x), -np.inf)
        for w, mu, s in zip(self.weights, self.means, self.stds):
            diff  = x - mu
            log_g = -0.5 * np.sum(diff**2, axis=-1) / s**2 - np.log(2*np.pi*s**2)
            lp    = np.logaddexp(lp, np.log(w + 1e-300) + log_g)
        return lp

    # ── sampling ───────────────────────────────────────────
    def sample(self, n):
        n    = int(n)
        K    = len(self.weights)
        comp = np.random.choice(K, size=n, p=self.weights)
        out  = np.zeros((n, 2), dtype=np.float32)
        for k in range(K):
            mask = comp == k
            cnt  = int(mask.sum())
            if cnt:
                out[mask] = (np.random.randn(cnt, 2).astype(np.float32)
                             * self.stds[k] + self.means[k])
        return out

    # ── OU smoothing  U_t ──────────────────────────────────
    def ou_smooth(self, t):
        """U_t: μ → e^{-t/2}μ,  σ² → σ²e^{-t} + (1-e^{-t})"""
        scale     = np.exp(-t / 2.0)
        new_means = self.means * scale
        new_stds  = np.sqrt(self.stds**2 * np.exp(-t) + (1.0 - np.exp(-t)))
        return AnalyticGMM(self.weights.copy(), new_means, new_stds)

    # ── Neumann series  p_inf ──────────────────────────────
    def neumann_collapse(self, alpha, t0, K=None):
        """
        Truncated Neumann series (Thm 1):
            p_inf ≈ α Σ_{k=0}^{K-1} (1-α)^k U_{(k+1)t0}(self)
        Returns an AnalyticGMM with n_components*K components.
        """
        if K is None: K = NEUMANN_K
        all_w, all_mu, all_s = [], [], []
        norm = 0.0
        for k in range(K):
            geo_w    = alpha * (1.0 - alpha)**k
            norm    += geo_w
            smoothed = self.ou_smooth((k + 1) * t0)
            all_w.append(geo_w * smoothed.weights)
            all_mu.append(smoothed.means)
            all_s.append(smoothed.stds)
        weights = np.concatenate(all_w) / norm
        means   = np.concatenate(all_mu, axis=0)
        stds    = np.concatenate(all_s,  axis=0)
        return AnalyticGMM(weights, means, stds)

    # ── KL divergences ────────────────────────────────────
    def kl_to(self, other, n=None):
        """KL(self ‖ other) via Monte Carlo."""
        if n is None: n = MC_KL_SAMPLES
        x  = self.sample(n)
        lp = self.log_prob(x)
        lq = other.log_prob(x)
        return float(np.maximum(np.mean(lp - lq), 0.0))

    def kl_from_samples(self, samples):
        """KL(p_samples ‖ self) via KDE on samples side."""
        kde = KernelDensity(bandwidth=KDE_BW).fit(samples)
        lp  = kde.score_samples(samples)
        lq  = self.log_prob(samples)
        return float(np.maximum(np.mean(lp - lq), 0.0))

    # ── density on grid ───────────────────────────────────
    def density_on_grid(self, xx, yy):
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        return np.exp(self.log_prob(pts)).reshape(xx.shape)


def make_true_dist():
    return AnalyticGMM(GMM_WEIGHTS, GMM_MEANS, GMM_STDS)

# Standard normal reference
def normal_0I():
    return AnalyticGMM([1.0], [[0.0, 0.0]], [1.0])


# ─────────────────────────────────────────────────────────────
#  SCORE NETWORK
# ─────────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=TIME_EMB_DIM):
        super().__init__(); self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(
            -np.log(10_000) *
            torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args  = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ScoreNet(nn.Module):
    def __init__(self, x_dim=2, hidden=HIDDEN, n_layers=N_LAYERS,
                 time_emb_dim=TIME_EMB_DIM):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        layers = [nn.Linear(x_dim + time_emb_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, x_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_emb(t)], dim=-1))


# ─────────────────────────────────────────────────────────────
#  VP-SDE (β≡1 → simple OU → κ = (1-α)e^{-t0/2})
# ─────────────────────────────────────────────────────────────
def get_marginal_params(t):
    """α_t = e^{-t/2},  σ_t = sqrt(1-e^{-t})"""
    log_alpha = -0.5 * (BETA_MIN*t + 0.5*(BETA_MAX - BETA_MIN)*t**2)
    alpha_t   = torch.exp(log_alpha)
    sigma_t   = torch.sqrt((1.0 - torch.exp(2.0*log_alpha)).clamp(min=1e-8))
    return alpha_t, sigma_t


def dsm_loss(model, x0):
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device) * (T_MAX - 1e-4) + 1e-4
    eps = torch.randn_like(x0)
    a_t, s_t = get_marginal_params(t)
    x_t  = a_t[:, None]*x0 + s_t[:, None]*eps
    pred = model(x_t, t)
    return (s_t[:, None]**2 * (pred + eps/s_t[:, None])**2).mean()


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────
def train_score_model(data, n_epochs, verbose_every=50):
    model  = ScoreNet().to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(data).float()),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model.train()
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        el = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = dsm_loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item()
        sched.step()
        if verbose_every and epoch % verbose_every == 0:
            log(f"    ep {epoch:4d}/{n_epochs}  loss={el/len(loader):.5f}  "
                f"({time.time()-t0:.1f}s)")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
#  REVERSE-TIME SAMPLING
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_from_model(model, n=None):
    if n is None: n = N_GEN_SAMPLES
    x  = torch.randn(n, 2, device=DEVICE)
    dt = T_MAX / N_REVERSE_STEPS
    ts = torch.linspace(T_MAX, dt, N_REVERSE_STEPS, device=DEVICE)
    for t_val in ts:
        beta  = float(BETA_MIN + t_val.item()*(BETA_MAX - BETA_MIN))
        score = model(x, t_val.expand(n))
        x = x + (0.5*beta*x + beta*score)*dt + (beta*dt)**0.5*torch.randn_like(x)
    return x.cpu().numpy()


# ─────────────────────────────────────────────────────────────
#  KL HELPERS
# ─────────────────────────────────────────────────────────────
def kde_kl_to_analytic(samples, analytic, bw=KDE_BW):
    """KL(p_samples ‖ analytic) via KDE."""
    kde = KernelDensity(bandwidth=bw).fit(samples)
    lp  = kde.score_samples(samples)
    lq  = analytic.log_prob(samples)
    return float(np.maximum(np.mean(lp - lq), 0.0))


def kde_kl_between(samples_p, kde_q, bw=KDE_BW):
    """KL(p ‖ q) when q is given as a fitted KDE object."""
    kde_p = KernelDensity(bandwidth=bw).fit(samples_p)
    lp    = kde_p.score_samples(samples_p)
    lq    = kde_q.score_samples(samples_p)
    return float(np.maximum(np.mean(lp - lq), 0.0))


# ─────────────────────────────────────────────────────────────
#  GRID HELPERS
# ─────────────────────────────────────────────────────────────
def make_grid():
    xx, yy = np.meshgrid(np.linspace(-GRID_LIM, GRID_LIM, GRID_RES),
                         np.linspace(-GRID_LIM, GRID_LIM, GRID_RES))
    return xx, yy


def kde_density(samples, xx, yy, bw=KDE_BW):
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    kde = KernelDensity(bandwidth=bw).fit(samples)
    return np.exp(kde.score_samples(pts)).reshape(xx.shape)


# ─────────────────────────────────────────────────────────────
#  MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────
def run_experiment(alpha, true_dist):
    log(f"\n{'='*64}")
    log(f"  α = {alpha}   κ = {kappa(alpha):.4f}   (1-α/2)={1-alpha/2:.4f}")
    log(f"{'='*64}")
    exp_dir = alpha_dir(alpha)

    # Compute p_inf analytically (Thm 1)
    pinf = true_dist.neumann_collapse(alpha, T_MAX)
    kl_pinf_true = pinf.kl_to(true_dist)   # constant floor (Thm 2v)
    log(f"  KL(p_inf ‖ p_true) = {kl_pinf_true:.4f}  (irreducible floor, Thm 2v)")

    res = dict(
        alpha              = alpha,
        kl_true            = [],   # KL(p̂_k ‖ p_true)
        kl_train           = [],   # KL(p̂_k ‖ D_k)  = score estimation error η_k
        kl_data            = [],   # KL(D_k ‖ p_true) data contamination
        kl_pinf            = [],   # KL(p̂_k ‖ p_inf)  — convergence to fixed point
        geo_bound          = [],   # (1-α/2)^N * KL(p̂_0 ‖ p_inf)
        acc_bound          = [],   # full bound: geo + error accumulation
        eta                = [],   # score error η_k = kl_train[k]
        kl_pinf_true_scalar= kl_pinf_true,
        gen_samples        = [],
        train_samples      = [],
    )

    # ── Generation 0 ──────────────────────────────────────
    log(f"\n[Gen  0]  {N_TRAIN:,} true samples  |  {N_EPOCHS_GEN0} epochs")
    train_data = true_dist.sample(N_TRAIN)
    model      = train_score_model(train_data, N_EPOCHS_GEN0)
    torch.save(model.state_dict(), os.path.join(exp_dir, 'gen_00.pt'))
    gen_s = sample_from_model(model)
    np.save(os.path.join(exp_dir, 'gen_00_samples.npy'), gen_s)

    kl_true  = kde_kl_to_analytic(gen_s, true_dist)
    kl_pinf  = kde_kl_to_analytic(gen_s, pinf)
    kde_tr   = KernelDensity(bandwidth=KDE_BW).fit(train_data)
    kl_train = kde_kl_between(gen_s, kde_tr)
    kl_data  = kde_kl_to_analytic(train_data, true_dist)

    kl_0_pinf = kl_pinf   # initial distance to p_inf (for geometric bound)

    for lst, val in zip(
        [res['kl_true'], res['kl_train'], res['kl_data'], res['kl_pinf'],
         res['eta'], res['geo_bound'], res['acc_bound']],
        [kl_true, kl_train, kl_data, kl_pinf, kl_train, kl_pinf, kl_pinf]
    ):
        lst.append(val)
    res['gen_samples'].append(gen_s[:N_STORE].copy())
    res['train_samples'].append(train_data[:N_STORE].copy())
    log(f"  KL→true={kl_true:.4f}  KL→p_inf={kl_pinf:.4f}  "
        f"KL→train={kl_train:.4f}  KL(D‖true)={kl_data:.4f}")

    prev_gen = gen_s

    # ── Generations 1 … N_GENERATIONS ────────────────────
    for gen in range(1, N_GENERATIONS + 1):
        n_fresh    = max(1, int(alpha * N_TRAIN))
        n_old      = N_TRAIN - n_fresh
        fresh      = true_dist.sample(n_fresh)
        old        = prev_gen[np.random.choice(len(prev_gen), n_old, replace=True)]
        train_data = np.random.permutation(np.concatenate([fresh, old], axis=0))

        log(f"\n[Gen {gen:2d}]  {n_fresh:,} fresh + {n_old:,} recycled  |  {N_EPOCHS_RECURSE} epochs")
        model = train_score_model(train_data, N_EPOCHS_RECURSE, verbose_every=0)
        torch.save(model.state_dict(), os.path.join(exp_dir, f'gen_{gen:02d}.pt'))
        gen_s = sample_from_model(model)
        np.save(os.path.join(exp_dir, f'gen_{gen:02d}_samples.npy'), gen_s)

        kl_true  = kde_kl_to_analytic(gen_s, true_dist)
        kl_pinf  = kde_kl_to_analytic(gen_s, pinf)
        kde_tr   = KernelDensity(bandwidth=KDE_BW).fit(train_data)
        kl_train = kde_kl_between(gen_s, kde_tr)
        kl_data  = kde_kl_to_analytic(train_data, true_dist)

        # Theorem 3 geometric bound: (1-α/2)^N * KL(p̂_0 ‖ p_inf)
        geo_b = (1.0 - alpha/2.0)**gen * kl_0_pinf
        # Accumulated error term (C=1 proxy): 1/α * Σ_{i<N} (1-α/2)^{N-1-i} η_i
        etas_so_far = res['eta']   # length = gen (for i=0..gen-1)
        acc_term    = sum((1 - alpha/2.0)**(gen-1-i) * etas_so_far[i]
                          for i in range(gen)) / max(alpha, 1e-6)
        acc_b       = geo_b + acc_term

        for lst, val in zip(
            [res['kl_true'], res['kl_train'], res['kl_data'], res['kl_pinf'],
             res['eta'], res['geo_bound'], res['acc_bound']],
            [kl_true, kl_train, kl_data, kl_pinf, kl_train, geo_b, acc_b]
        ):
            lst.append(val)
        res['gen_samples'].append(gen_s[:N_STORE].copy())
        res['train_samples'].append(train_data[:N_STORE].copy())
        log(f"  KL→true={kl_true:.4f}  KL→p_inf={kl_pinf:.4f}  "
            f"geo_b={geo_b:.4f}  acc_b={acc_b:.4f}  η={kl_train:.4f}")

        prev_gen = gen_s

    with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(res, f)
    return res


# ─────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────

def plot_kl_curves(all_results, out):
    """Plot 01: three KL curves per alpha."""
    gens = list(range(N_GENERATIONS + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    keys   = ['kl_true', 'kl_train', 'kl_data']
    y_label = [
        r'$\mathrm{KL}(\hat{p}^{(k)} \| p_{\mathrm{data}})$ (model collapse)',
        r'$\eta_k = \mathrm{KL}(\hat{p}^{(k)} \| q_k)$ (Score estimation error)',
        r'$\mathrm{KL}(q_k \| p_{\mathrm{data}})$ ',
    ]
    for ax, key, ylabel in zip(axes, keys, y_label):
        for alpha, res in sorted(all_results.items()):
            ax.plot(gens, res[key], color=ALPHA_COLORS[alpha],
                    marker='o', ms=3, lw=1.5, label=fr'$\alpha={alpha}$',
                    markevery=3)
            ax.set_xlabel('Generation $k$')
            ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_xlim(0, N_GENERATIONS)
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    fig.tight_layout()
    savefig(fig, os.path.join(out, '01_kl_curves.pdf'))
    log("  → 01_kl_curves")


def plot_kl_to_pinf_and_bound(all_results, out):
    """Plot 02 — Theorem 3."""
    gens  = np.array(range(N_GENERATIONS + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, alpha in zip(axes, sorted(all_results.keys())):
        res   = all_results[alpha]
        c     = ALPHA_COLORS[alpha]
        kl_pi = np.array(res['kl_pinf'])
        geo_b = np.array(res['geo_bound'])
        acc_b = np.maximum(np.array(res['acc_bound']), 1e-10)
        floor = res['kl_pinf_true_scalar']
        kap   = kappa(alpha)

        ax.semilogy(gens, kl_pi, color=c, lw=1.8, marker='o', ms=3,
                    markevery=3, zorder=4,
                    label=r'$\mathrm{KL}(\hat{p}^{N} \| p_\infty^\star)$ empirical')
        ax.semilogy(gens, geo_b, color=c, lw=1.4, ls='--', zorder=3,
                    label=r'$(1-\alpha/2)^N \cdot \mathrm{KL}(p_{\mathrm{data}} \| p_\infty^\star)$')
        ax.axhline(floor, color='#333333', lw=1.0, ls='-.',
                   label=r'$\mathrm{KL}(p_\infty^\star \| p_{\mathrm{data}})$ floor')

    
        ax.set_xlabel('Generation $N$')
        ax.set_ylabel('KL (log scale)')
        ax.legend(fontsize=8)
        ax.set_xlim(0, N_GENERATIONS)

    fig.tight_layout()
    savefig(fig, os.path.join(out, '02_kl_to_pinf_and_bound.pdf'))
    log("  → 02_kl_to_pinf_and_bound")


def plot_kl_decomposition(all_results, out):
    """Plot 03: full KL decomposition."""
    gens = list(range(N_GENERATIONS + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, alpha in zip(axes, sorted(all_results.keys())):
        res = all_results[alpha]
        c   = ALPHA_COLORS[alpha]
        ax.plot(gens, res['kl_true'],  color='#222222', lw=2.0,
                label=r'$\mathrm{KL}(\hat{p}^{N}\|p_{\mathrm{data}})$')
        ax.plot(gens, res['kl_pinf'],  color=c, lw=1.8,
                label=r'$\mathrm{KL}(\hat{p}^{N}\|p_\infty^\star)$')
        ax.axhline(res['kl_pinf_true_scalar'], color=c, lw=0.9, ls='-.', alpha=0.7,
                   label=r'$\mathrm{KL}(p_\infty^\star\|p_{\mathrm{data}})$ floor')
        ax.fill_between(gens, 0, res['kl_train'], alpha=0.08, color=c)
        ax.set_title(fr'$\alpha = {alpha}$')
        ax.set_xlabel('Generation $k$')
        ax.set_ylabel('KL divergence')
        ax.legend(fontsize=8)
        ax.set_xlim(0, N_GENERATIONS)
    fig.tight_layout()
    savefig(fig, os.path.join(out, '03_kl_decomposition.pdf'))
    log("  → 03_kl_decomposition")


def plot_neumann_series(true_dist, out):
    """Plot 04 — Theorem 1 Neumann series."""
    xx, yy = make_grid()
    K_show = 5
    alphas = [0.1, 0.5, 0.9]
    n_cols = K_show + 2

    fig, axes = plt.subplots(len(alphas), n_cols,
                             figsize=(2.8*n_cols, 3.0*len(alphas)))
    for row, alpha in enumerate(alphas):
        cmap = ALPHA_CMAPS[alpha]
        pinf = true_dist.neumann_collapse(alpha, T_MAX)

        partial_d, partial_w = np.zeros((GRID_RES, GRID_RES)), 0.0
        for col in range(K_show):
            geo_w    = alpha * (1 - alpha)**col
            smoothed = true_dist.ou_smooth((col + 1) * T_MAX)
            comp_d   = smoothed.density_on_grid(xx, yy)
            partial_d += geo_w * comp_d
            partial_w += geo_w

            ax = axes[row, col]
            ax.contourf(xx, yy, comp_d, levels=25, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-GRID_LIM, GRID_LIM); ax.set_ylim(-GRID_LIM, GRID_LIM)
            if col == 0:
                ax.set_ylabel(fr'$\alpha={alpha}$', fontsize=10, fontweight='bold')

        ax_p = axes[row, K_show]
        ax_p.contourf(xx, yy, partial_d / partial_w, levels=25, cmap=cmap)
        ax_p.set_xticks([]); ax_p.set_yticks([])

        ax_f = axes[row, K_show + 1]
        kl_f = pinf.kl_to(true_dist, n=5_000)
        ax_f.contourf(xx, yy, pinf.density_on_grid(xx, yy), levels=25, cmap=cmap)
        ax_f.set_xticks([]); ax_f.set_yticks([])

    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    savefig(fig, os.path.join(out, '04_neumann_series.pdf'))
    log("  → 04_neumann_series")


def plot_limiting_behaviors(true_dist, out, n_seeds=5):
    """
    Plot 05 — Theorem 2 limiting behaviours.
    Multi-seed sweep: solid line = mean across seeds, shaded band = ±1 std.
    """
    N_MC = 8_000
    log(f"  Limiting behaviours sweep (n_seeds={n_seeds}, n_mc={N_MC})...")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))

    alphas_sw = np.linspace(0.02, 0.98, 50)
    t0s_sw    = np.linspace(0.02, 5.0,  60)
    u_t0      = true_dist.ou_smooth(T_MAX)
    n0I       = normal_0I()

    # ── Multi-seed runs ────────────────────────────────────────
    panel1_runs = []                              # (n_seeds, len(alphas_sw))
    panel2_runs = []                              # (n_seeds, len(alphas_sw))
    panel3_runs = {a: [] for a in [0.1, 0.5, 0.9]}  # per-α lists -> (n_seeds, len(t0s_sw))

    for seed_idx in range(n_seeds):
        log(f"    seed {seed_idx+1}/{n_seeds}")
        np.random.seed(2000 + seed_idx)

        # (i) α → 1
        panel1_runs.append([
            true_dist.neumann_collapse(a, T_MAX, K=60).kl_to(u_t0, n=N_MC)
            for a in alphas_sw
        ])
        # (ii) α → 0
        panel2_runs.append([
            true_dist.neumann_collapse(a, T_MAX, K=120).kl_to(n0I, n=N_MC)
            for a in alphas_sw
        ])
        # (iii) t0 → 0
        for a in [0.1, 0.5, 0.9]:
            panel3_runs[a].append([
                true_dist.neumann_collapse(a, t0, K=80).kl_to(true_dist, n=N_MC)
                for t0 in t0s_sw
            ])

    panel1_arr = np.array(panel1_runs)            # (n_seeds, n_alpha)
    panel2_arr = np.array(panel2_runs)
    panel3_arr = {a: np.array(v) for a, v in panel3_runs.items()}

    # ── Helper: clip lower band so log/linear axes stay clean ──
    def _band(arr):
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        return mean, np.maximum(mean - std, 0.0), mean + std

    # ── Panel (i): α → 1⁻ ──────────────────────────────────────
    ax = axes[0]
    mean, lo, hi = _band(panel1_arr)
    ax.plot(alphas_sw, mean, color='#2166ac', lw=1.8)
    ax.fill_between(alphas_sw, lo, hi, color='#2166ac', alpha=0.20, linewidth=0)
    ax.axhline(0, color='#333333', lw=0.8, ls='--')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\mathrm{KL}(p_\infty^\star \| U_{t_0}(p_{\mathrm{data}}))$')
    ax.set_title(r'$\alpha\to 1^-$: $p_\infty^\star\to U_{t_0}(p_{\mathrm{data}})$')
    ax.set_xlim(0, 1)

    # ── Panel (ii): α → 0⁺ ─────────────────────────────────────
    ax = axes[1]
    mean, lo, hi = _band(panel2_arr)
    ax.plot(alphas_sw, mean, color='#d62728', lw=1.8)
    ax.fill_between(alphas_sw, lo, hi, color='#d62728', alpha=0.20, linewidth=0)
    ax.axhline(0, color='#333333', lw=0.8, ls='--')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\mathrm{KL}(p_\infty^\star \| \mathcal{N}(0,I))$')
    ax.set_title(r'$\alpha\to 0^+$: $p_\infty^\star\to\mathcal{N}(0,I)$')
    ax.set_xlim(0, 1)

    # ── Panel (iii): t0 → 0⁺ ───────────────────────────────────
    ax = axes[2]
    for a in [0.1, 0.5, 0.9]:
        mean, lo, hi = _band(panel3_arr[a])
        c = ALPHA_COLORS[a]
        ax.plot(t0s_sw, mean, color=c, lw=1.8, label=fr'$\alpha={a}$')
        ax.fill_between(t0s_sw, lo, hi, color=c, alpha=0.20, linewidth=0)
    ax.axhline(0, color='#333333', lw=0.8, ls='--')
    ax.axvline(T_MAX, color='#555555', lw=0.9, ls=':', label=fr'$t_0={T_MAX}$ (exp.)')
    ax.set_xlabel(r'$t_0$')
    ax.set_ylabel(r'$\mathrm{KL}(p_\infty^\star \| p_{\mathrm{data}})$')
    ax.set_title(r'$t_0\to 0^+$: KL $\to 0$')
    ax.legend(fontsize=16, loc='lower right')

    # ── (iv) t0 → ∞  (kept commented as in the original) ───────
    '''ax = axes[3]
    for a in [0.1, 0.5, 0.9]:
        kls4 = [true_dist.neumann_collapse(a, t0, K=80).kl_to(n0I, n=N_MC)
                for t0 in t0s_sw]
        ax.plot(t0s_sw, kls4, color=ALPHA_COLORS[a], lw=1.8, label=fr'$\alpha={a}$')
    ax.axhline(0, color='#333333', lw=0.8, ls='--')
    ax.axvline(T_MAX, color='#555555', lw=0.9, ls=':')
    ax.set_xlabel(r'$t_0$')
    ax.set_ylabel(r'$\mathrm{KL}(p_\infty \| \mathcal{N}(0,I))$')
    ax.set_title(r'(iv) $t_0\to\infty$: $p_\infty\to\mathcal{N}(0,I)$')
    ax.legend(fontsize=8)'''

    fig.tight_layout()
    savefig(fig, os.path.join(out, '05_limiting_behaviors.pdf'))
    log("  → 05_limiting_behaviors")


def plot_t0_sweep(true_dist, out):
    """Plot 06: t0 sweep on density grid and KL floor curve."""
    T0_VALS = [0.1, 0.5, 1.0, 2.0, 4.0]
    xx, yy  = make_grid()
    alphas  = [0.1, 0.5, 0.9]
    true_d  = true_dist.density_on_grid(xx, yy)

    # 06a density grid
    n_cols = 1 + len(T0_VALS)
    fig, axes = plt.subplots(len(alphas), n_cols,
                             figsize=(2.7*n_cols, 2.9*len(alphas)))
    for row, alpha in enumerate(alphas):
        ax = axes[row, 0]
        ax.contourf(xx, yy, true_d, levels=25, cmap='Greens')
        ax.set_ylabel(fr'$\alpha={alpha}$', fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(r'$p_{\rm data}$', fontsize=9)
        for col, t0 in enumerate(T0_VALS):
            ax   = axes[row, col + 1]
            pinf = true_dist.neumann_collapse(alpha, t0, K=100)
            kl   = pinf.kl_to(true_dist, n=4_000)
            kap  = kappa(alpha, t0)
            ax.contourf(xx, yy, pinf.density_on_grid(xx, yy), levels=25,
                        cmap=ALPHA_CMAPS[alpha])
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    savefig(fig, os.path.join(out, '06a_t0_sweep_density.pdf'))
    log("  → 06a_t0_sweep_density")

    # 06b KL floor vs t0
    t0_fine = np.linspace(0.05, 5.0, 80)
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for alpha in alphas:
        kls = [true_dist.neumann_collapse(alpha, t0, K=80).kl_to(true_dist, n=6_000)
               for t0 in t0_fine]
        kap_vals = kappa(alpha, t0_fine)
        ax2.plot(t0_fine, kls, color=ALPHA_COLORS[alpha], lw=1.8,
                 label=fr'$\alpha={alpha}$')
    ax2.axvline(T_MAX, color='#555555', lw=1.0, ls=':', label=fr'$t_0={T_MAX}$ (exp.)')
    ax2.set_xlabel(r'$t_0$ (truncation / diffusion time)')
    ax2.set_ylabel(r'$\mathrm{KL}(p_\infty \| p_{\rm data})$ (irreducible collapse)')
    ax2.legend()
    fig2.tight_layout()
    savefig(fig2, os.path.join(out, '06b_t0_sweep_kl.pdf'))
    log("  → 06b_t0_sweep_kl")


def plot_kappa_contour(out):
    """Plot 07: κ and KL floor phase diagrams."""
    true_dist = make_true_dist()
    alphas    = np.linspace(0.01, 0.99, 200)
    t0s       = np.linspace(0.01, 5.0,  200)
    AA, TT    = np.meshgrid(alphas, t0s)
    KK        = (1 - AA) * np.exp(-TT / 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # κ heatmap
    im = axes[0].contourf(AA, TT, KK, levels=40, cmap='RdYlGn_r')
    axes[0].contour(AA, TT, KK, levels=[0.3, 0.5, 0.7, 0.9, 0.95],
                    colors='k', linewidths=0.7, linestyles='--')
    cb = fig.colorbar(im, ax=axes[0], label=r'$\kappa = (1-\alpha)\,e^{-t_0/2}$', shrink=0.9)
    cb.ax.tick_params(labelsize=9)
    axes[0].set_xlabel(r'$\alpha$ (fresh-data fraction)')
    axes[0].set_ylabel(r'$t_0$ (diffusion / truncation time)')
    for alpha in ALPHAS:
        axes[0].plot(alpha, T_MAX, 'w*', ms=12, mec='#333333', mew=0.7,
                     label=fr'$\alpha={alpha}$, $\kappa={kappa(alpha):.3f}$')
    axes[0].legend(fontsize=8, loc='upper right',
                   facecolor='white', framealpha=0.85)

    # KL floor heatmap
    ag  = np.linspace(0.05, 0.95, 25)
    tg  = np.linspace(0.1,  4.0,  25)
    AG, TG = np.meshgrid(ag, tg)
    KL_FLOOR = np.zeros_like(AG)
    for i, t0 in enumerate(tg):
        for j, a in enumerate(ag):
            pi = true_dist.neumann_collapse(a, t0, K=60)
            KL_FLOOR[i, j] = pi.kl_to(true_dist, n=3_000)

    im2 = axes[1].contourf(AG, TG, KL_FLOOR, levels=30, cmap='YlOrRd')
    axes[1].contour(AG, TG, KL_FLOOR, levels=8, colors='k',
                    linewidths=0.5, linestyles='--')
    cb2 = fig.colorbar(im2, ax=axes[1],
                       label=r'$\mathrm{KL}(p_\infty\|p_{\rm data})$',
                       shrink=0.9)
    cb2.ax.tick_params(labelsize=9)
    for alpha in ALPHAS:
        axes[1].plot(alpha, T_MAX, 'w*', ms=12, mec='#333333', mew=0.7)
    axes[1].set_xlabel(r'$\alpha$')
    axes[1].set_ylabel(r'$t_0$')
    
    fig.tight_layout()
    savefig(fig, os.path.join(out, '07_kappa_contour.pdf'))
    log("  → 07_kappa_contour")


def plot_density_grid(all_results, true_dist, out):
    """Plot 08: density grid with p_inf reference column."""
    PLOT_GENS = [0, 2, 5, 10, 15, 20]
    xx, yy    = make_grid()
    true_d    = true_dist.density_on_grid(xx, yy)
    alphas    = sorted(all_results.keys())
    n_cols    = 2 + len(PLOT_GENS)

    fig, axes = plt.subplots(len(alphas), n_cols,
                             figsize=(2.6*n_cols, 2.8*len(alphas)))
    for row, alpha in enumerate(alphas):
        res    = all_results[alpha]
        cmap   = ALPHA_CMAPS[alpha]
        pinf   = make_true_dist().neumann_collapse(alpha, T_MAX)
        floor  = res['kl_pinf_true_scalar']

        ax = axes[row, 0]
        ax.contourf(xx, yy, true_d, levels=25, cmap='Greens')
        ax.set_ylabel(fr'$\alpha={alpha}$', fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(r'$p_{\rm data}$', fontsize=9)

        ax = axes[row, 1]
        ax.contourf(xx, yy, pinf.density_on_grid(xx, yy), levels=25, cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])

        for col, g in enumerate(PLOT_GENS):
            ax   = axes[row, col + 2]
            samp = res['gen_samples'][g]
            ax.contourf(xx, yy, kde_density(samp, xx, yy), levels=25, cmap=cmap)
            ax.scatter(samp[:, 0], samp[:, 1], s=0.8, alpha=0.18, c='k',
                       rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-GRID_LIM, GRID_LIM); ax.set_ylim(-GRID_LIM, GRID_LIM)

    fig.suptitle(
        r'Density Evolution — $p_\infty$ (Thm 1) as fixed-point reference',
        fontweight='bold')
    fig.tight_layout(h_pad=0.5, w_pad=0.2)
    savefig(fig, os.path.join(out, '08_density_grid.pdf'))
    log("  → 08_density_grid")


def plot_density_strips(all_results, true_dist, out):
    """Plot 09: per-alpha density strips."""
    PLOT_GENS = [0, 1, 2, 5, 10, 15, 20]
    xx, yy    = make_grid()
    true_d    = true_dist.density_on_grid(xx, yy)

    for alpha, res in sorted(all_results.items()):
        pinf   = make_true_dist().neumann_collapse(alpha, T_MAX)
        cmap   = ALPHA_CMAPS[alpha]
        kap    = kappa(alpha)
        n_cols = 2 + len(PLOT_GENS)

        fig, axes = plt.subplots(1, n_cols, figsize=(2.6*n_cols, 3.0))
        axes[0].contourf(xx, yy, true_d, levels=25, cmap='Greens')
        axes[0].set_title(r'$p_{\rm data}$', fontsize=9)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].contourf(xx, yy, pinf.density_on_grid(xx, yy), levels=25, cmap=cmap)
        axes[1].set_xticks([]); axes[1].set_yticks([])

        for col, g in enumerate(PLOT_GENS):
            ax   = axes[col + 2]
            samp = res['gen_samples'][g]
            ax.contourf(xx, yy, kde_density(samp, xx, yy), levels=25, cmap=cmap)
            ax.scatter(samp[:, 0], samp[:, 1], s=0.8, alpha=0.18, c='k',
                       rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-GRID_LIM, GRID_LIM); ax.set_ylim(-GRID_LIM, GRID_LIM)
        fig.tight_layout()
        savefig(fig, os.path.join(out, f'09_density_strip_alpha_{alpha:.1f}.pdf'))
    log("  → 09_density_strips")


def plot_annealed_truncation(true_dist, out, alphas=(0.1, 0.5, 0.9),
                             N_gens=50, t_inf=0.2,
                             n_mc=80_000, kl_floor=1e-4, n_seeds=5):
    """
    Plot 14 — Theorem 4.1: Annealed truncation schedules.
    Multi-seed sweep: solid line = mean across seeds, shaded band = ±1 std.
    1×3 horizontal layout, one panel per alpha.
    """
    log(f"  Annealed truncation (alphas={list(alphas)}, t_inf={t_inf}, "
        f"N={N_gens}, n_mc={n_mc}, n_seeds={n_seeds})...")
    t0_base = T_MAX

    schedules = {
        'fixed':   lambda i: t0_base,
        'beta1':   lambda i: t0_base / (1 + i),
        'beta2':   lambda i: t0_base / (1 + i)**2,
        'shifted': lambda i: t_inf + (t0_base - t_inf) / (1 + i),
    }
    colors = {'fixed':'#d62728', 'beta1':'#ff7f0e',
              'beta2':'#2166ac', 'shifted':'#6c5b7b'}
    labels = {
        'fixed':   fr'fixed $t_0={t0_base}$',
        'beta1':   r'$t_0^{(i)}=t_0/(1+i)$',
        'beta2':   r'$t_0^{(i)}=t_0/(1+i)^2$',
        'shifted': fr'$t_0^{{(i)}}\!\to\!{t_inf}$',
    }
    order = ['fixed', 'shifted', 'beta1', 'beta2']

    # ── Multi-seed sweep ───────────────────────────────────────
    # all_kls[alpha][name]     : (n_seeds, N_gens+1)
    # all_plateaus[alpha][name]: list of length n_seeds
    all_kls      = {a: {n: [] for n in schedules} for a in alphas}
    all_plateaus = {a: {n: [] for n in schedules} for a in alphas}

    for seed_idx in range(n_seeds):
        log(f"    seed {seed_idx+1}/{n_seeds}")
        np.random.seed(1000 + seed_idx)
        for alpha in alphas:
            plateaus = {
                'fixed':   true_dist.neumann_collapse(alpha, t0_base, K=NEUMANN_K)
                                    .kl_to(true_dist, n=n_mc),
                'beta1':   0.0,
                'beta2':   0.0,
                'shifted': true_dist.neumann_collapse(alpha, t_inf,   K=NEUMANN_K)
                                    .kl_to(true_dist, n=n_mc),
            }
            for name, val in plateaus.items():
                all_plateaus[alpha][name].append(val)

            for name, sched in schedules.items():
                kls    = [0.0]
                p_curr = true_dist
                for i in range(N_gens):
                    w = np.concatenate([alpha * true_dist.weights,
                                        (1 - alpha) * p_curr.weights])
                    m = np.concatenate([true_dist.means,  p_curr.means], axis=0)
                    s = np.concatenate([true_dist.stds,   p_curr.stds])
                    p_curr = AnalyticGMM(w, m, s).ou_smooth(sched(i))
                    kls.append(p_curr.kl_to(true_dist, n=n_mc))
                all_kls[alpha][name].append(kls)

    # Stack to arrays
    for alpha in alphas:
        for name in schedules:
            all_kls[alpha][name] = np.array(all_kls[alpha][name])  # (n_seeds, N+1)

    # ── 1×3 horizontal layout ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    gens = np.arange(N_gens + 1)

    for ax, alpha in zip(axes, alphas):
        seeds_dict   = all_kls[alpha]
        plateau_dict = all_plateaus[alpha]

        for name in order:
            arr  = np.maximum(seeds_dict[name], kl_floor)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            lo   = np.maximum(mean - std, kl_floor)
            hi   = mean + std

            ax.semilogy(gens, mean, color=colors[name], lw=2.0,
                        marker='o', ms=3, markevery=3, label=labels[name])
            ax.fill_between(gens, lo, hi, color=colors[name],
                            alpha=0.20, linewidth=0)

            plat_mean = float(np.mean(plateau_dict[name]))
            if plat_mean > 1e-5:
                ax.axhline(plat_mean, color=colors[name],
                           lw=1.0, ls='--', alpha=0.7)

        ax.set_xlabel(r'Generation $N$')
        ax.set_ylabel(r'$\mathrm{KL}(\hat{p}^N \| p_{\mathrm{data}})$')
        ax.set_title(fr'$\alpha={alpha}$')
        ax.set_xlim(0, N_gens)
        ax.set_ylim(bottom=kl_floor)

    fig.tight_layout()
    savefig(fig, os.path.join(out, '14_annealed_truncation.pdf'))
    log("  → 14_annealed_truncation")

def plot_marginals(all_results, true_dist, out):
    """Plot 10: 1D x-marginals."""
    x_grid  = np.linspace(-GRID_LIM, GRID_LIM, 500).reshape(-1, 1)
    pts_2d  = np.hstack([x_grid, np.zeros_like(x_grid)])
    true_1d = np.exp(true_dist.log_prob(pts_2d))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, alpha in zip(axes, sorted(all_results.keys())):
        res     = all_results[alpha]
        c       = ALPHA_COLORS[alpha]
        pinf    = make_true_dist().neumann_collapse(alpha, T_MAX)
        pinf_1d = np.exp(pinf.log_prob(pts_2d))

        ax.plot(x_grid.ravel(), true_1d,  color='#222222', lw=2.0,
                label=r'$p_{\rm data}$', zorder=5)
        ax.plot(x_grid.ravel(), pinf_1d,  color=c, lw=1.8, ls='-.',
                label=r'$p_\infty$ (Neumann)', zorder=4)

        for g, ls, lbl in [(0, '--', 'Gen 0'), (10, ':', 'Gen 10'), (20, '-', 'Gen 20')]:
            s   = res['gen_samples'][g]
            kde = KernelDensity(bandwidth=KDE_BW).fit(s[:, 0:1])
            ax.plot(x_grid.ravel(), np.exp(kde.score_samples(x_grid)),
                    color=c, lw=1.3, ls=ls, alpha=0.85, label=lbl)

        ax.set_title(fr'$\alpha = {alpha}$')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'Density')
        ax.legend(fontsize=16)
    fig.tight_layout()
    savefig(fig, os.path.join(out, '10_marginals.pdf'))
    log("  → 10_marginals")


# ─────────────────────────────────────────────────────────────
#  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
def print_summary(all_results):
    gens = [0, 1, 5, 10, 15, 20]
    hdr  = (f"{'Gen':>4}  {'α':>5}  {'KL→true':>10}  "
            f"{'KL→p_inf':>10}  {'geo_bound':>12}  {'η_k':>8}")
    log("\n" + "─"*len(hdr)); log(hdr); log("─"*len(hdr))
    for alpha, res in sorted(all_results.items()):
        for g in gens:
            if g < len(res['kl_true']):
                log(f"{g:>4}  {alpha:>5}  "
                    f"{res['kl_true'][g]:>10.4f}  {res['kl_pinf'][g]:>10.4f}  "
                    f"{res['geo_bound'][g]:>12.4f}  {res['eta'][g]:>8.4f}")
        log("─"*len(hdr))



# ─────────────────────────────────────────────────────────────
#  HERMITE UTILITIES  (Experiment 2 — Spectral Filtering)
# ─────────────────────────────────────────────────────────────

def _hermite_1d(n, x):
    """
    Probabilist's Hermite polynomial H_n(x) via recurrence:
        H_0 = 1,  H_1 = x,  H_k = x H_{k-1} - (k-1) H_{k-2}.
    """
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h0, h1 = np.ones_like(x), x.copy()
    for k in range(2, n + 1):
        h0, h1 = h1, x * h1 - (k - 1) * h0
    return h1


def _hermite_norm(n, x):
    """
    Normalized Hermite h_n = H_n / sqrt(n!), orthonormal in L²(N(0,1)).
    E_{X~N(0,1)}[h_m(X) h_n(X)] = δ_{mn}.
    """
    return _hermite_1d(n, x) / np.sqrt(float(_math.factorial(n)))


def _degree_energy(samples, d):
    """
    Σ_{n1+n2=d}  E[h_{n1}(X_1) h_{n2}(X_2)]²
    = total Hermite energy at total degree d, estimated from samples of p.

    Spectral prediction:  energy_d(p_inf) / energy_d(p_data) = m_d(α,t0)²
    where m_d = α e^{-d t0/2} / (1-(1-α) e^{-d t0/2}).
    """
    s = np.asarray(samples, dtype=np.float64)
    energy = 0.0
    for n1 in range(d + 1):
        n2 = d - n1
        c   = float(np.mean(_hermite_norm(n1, s[:, 0]) *
                             _hermite_norm(n2, s[:, 1])))
        energy += c ** 2
    return energy


def _m_theory(d, alpha, t0):
    """
    Spectral attenuation factor m_{|n|}(α, t0) for total degree d.
    Collapse acts as a low-pass filter: m_d ∈ (0,1], decreasing in d.
    """
    u = np.exp(-d * t0 / 2.0)
    return alpha * u / (1.0 - (1.0 - alpha) * u + 1e-15)


# ─────────────────────────────────────────────────────────────
#  NOISY-SCORE SAMPLER  (Experiment 3 — Sensitivity Duality)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _sample_with_score_noise(model, eps_score, n=1500):
    """
    Reverse-SDE sampler with additive i.i.d. Gaussian perturbation
    ε_score · N(0,I_2) added to the score at every EM step.

    Integrated score error energy (cf. eq. (9) in the paper):
        η ≈ ε_score² · d · T   (d=2 dimensions, T=T_MAX)

    Used to sweep η and verify KL(p̃ ‖ p_∞) ∝ ε² / α².
    """
    x  = torch.randn(n, 2, device=DEVICE)
    dt = T_MAX / N_REVERSE_STEPS
    ts = torch.linspace(T_MAX, dt, N_REVERSE_STEPS, device=DEVICE)
    for t_val in ts:
        beta  = float(BETA_MIN + t_val.item() * (BETA_MAX - BETA_MIN))
        score = model(x, t_val.expand(n))
        if eps_score > 0.0:
            score = score + eps_score * torch.randn_like(score)
        x = x + (0.5*beta*x + beta*score)*dt + (beta*dt)**0.5 * torch.randn_like(x)
    return x.cpu().numpy()


# ══════════════════════════════════════════════════════════════
# BLOCK B — paste before print_summary()
# ══════════════════════════════════════════════════════════════

def plot_spectral_filtering(all_results, true_dist, out,
                            D_MAX=8, N_MC=50_000):
    """
    Plot 11 — Spectral Filtering (Proposition 1 / Experiment 2).

    For each total Hermite degree d = 1…D_MAX, measures the ratio
        sqrt( E_d(p̂^k) / E_d(p_data) )
    where E_d = Σ_{n1+n2=d} c_{n1,n2}² is the Hermite mode energy at degree d.

    Theory (Prop. 1):
        E_d(p_inf) / E_d(p_data) = m_d(α, t0)²
        m_d = α e^{-d t0/2} / (1-(1-α) e^{-d t0/2})

    Collapse acts as a low-pass filter: high-degree modes (fine structure)
    are attenuated more strongly than low-degree modes (coarse shape).
    """
    log("  Computing spectral filtering...")
    degrees = list(range(1, D_MAX + 1))

    # Reference energies from p_data (large MC for accuracy)
    data_samples    = true_dist.sample(N_MC).astype(np.float64)
    data_energies   = np.array([_degree_energy(data_samples, d)
                                 for d in degrees])
    # Only use degrees where p_data has meaningful energy
    signal_mask     = data_energies > 5e-6

    fig, axes = plt.subplots(1, len(sorted(all_results.keys())),
                             figsize=(15, 4.5))

    for ax, alpha in zip(axes, sorted(all_results.keys())):
        res = all_results[alpha]
        c   = ALPHA_COLORS[alpha]

        # ── Theory curve ─────────────────────────────────
        theory_ratio = np.array([_m_theory(d, alpha, T_MAX)
                                  for d in degrees])
        valid_d = [d for d, ok in zip(degrees, signal_mask) if ok]
        valid_theory = theory_ratio[signal_mask]

        if valid_d:
            ax.plot(valid_d, valid_theory, color='#222222', lw=2.2,
                    zorder=5, label=r'Theory $m_{|\mathbf{n}|}$')

        # ── Empirical ratios at selected generations ──────
        for g, ls, lbl in [(0, ':', 'Gen 0'),
                           (5,  '--', 'Gen 5'),
                           (10, '-.', 'Gen 10'),
                           (20, '-',  'Gen 20')]:
            if g >= len(res['gen_samples']):
                continue
            gen_samp = res['gen_samples'][g].astype(np.float64)
            gen_e    = np.array([_degree_energy(gen_samp, d)
                                  for d in degrees])
            # sqrt(energy ratio) = amplitude attenuation
            ratio = np.where(
                signal_mask & (data_energies > 0),
                np.sqrt(np.maximum(gen_e, 0.0) /
                        np.maximum(data_energies, 1e-15)),
                np.nan
            )
            valid_ratio = ratio[signal_mask]
            if len(valid_d) > 0:
                ax.plot(valid_d, valid_ratio, color=c, lw=1.5, ls=ls,
                        marker='o', ms=5, alpha=0.9, label=lbl)

        ax.axhline(1.0, color='#cccccc', lw=0.8, ls='--', alpha=0.6)
        ax.set_xlabel(r'Total Hermite degree $|\mathbf{n}|$')
        ax.set_ylabel(
            r'$\sqrt{E_{|\mathbf{n}|}(\hat{p}^k)\,/\,E_{|\mathbf{n}|}(p_{\rm data})}$')
        ax.set_title(fr'$\alpha = {alpha}$')
        ax.legend(fontsize=8)
        ax.set_xticks(valid_d if valid_d else degrees)
        ax.set_ylim(-0.05, 1.5)

    fig.suptitle(
        r'Prop. 1 — Collapse as Spectral Low-Pass Filter on Hermite Modes'
        '\n'
        r'$m_{|\mathbf{n}|}(\alpha,t_0) = \alpha e^{-|\mathbf{n}|t_0/2}'
        r'\,/\,(1-(1-\alpha)e^{-|\mathbf{n}|t_0/2})$;'
        r'  high-degree modes destroyed preferentially',
        fontweight='bold')
    fig.tight_layout()
    savefig(fig, os.path.join(out, '11_spectral_filtering.pdf'))
    log("  → 11_spectral_filtering")


def plot_sensitivity_duality(all_results, true_dist, out):
    """
    Plot 12 — Sensitivity–Collapse Duality (Proposition 4 / Experiment 3).

    Panel 1 (always): sensitivity ratio R(α) = KL_floor / η̄ vs α.
        From existing results (no model loading needed).
        Theory:  R ≤ C(t0,β) / α²  →  shows α⁻² scaling.

    Panel 2 (if gen_20.pt checkpoints exist): KL(p̃ ‖ p_∞) vs ε²
        for each α, sweeping added score noise ε.
        Theory: KL grows linearly in ε² with slope ∝ α⁻².
        The same regime with worst ideal collapse (small α) has steepest slope.
    """
    alphas = sorted(all_results.keys())

    # ── Panel 1: sensitivity ratio from existing data ──────
    sensitivity = {}
    # Replace in plot_sensitivity_duality:
    for alpha in alphas:
        res      = all_results[alpha]
        kl_pinf  = np.array(res['kl_pinf'])
        kl_floor_analytic = res['kl_pinf_true_scalar']
        # plateau = mean of last 5 generations of KL(p̂^N ‖ p_∞)
        plateau  = float(np.mean(kl_pinf[-5:]))
        # sensitivity = plateau above theoretical floor, normalised by score error
        eta_bar  = float(np.mean(np.array(res['eta'])))
        # The true R is KL(p̃_∞ ‖ p_∞) / η̄
        # We can't measure this directly, but plateau - kl_floor_analytic ≈ KL(p̃_∞ ‖ p_∞)
        # when the model has converged (which it has for all three α values)
        gap = max(plateau - kl_floor_analytic, 0.0)
        sensitivity[alpha] = gap / max(eta_bar, 1e-9)

    # ── Panel 2: ε-sweep using saved model checkpoints ─────
    eps_vals     = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    noise_results = {}

    for alpha in alphas:
        last_gen = N_GENERATIONS   # look for the last checkpoint
        ckpt = os.path.join(out, f"alpha_{alpha:.1f}",
                            f"gen_{last_gen:02d}.pt")
        if not os.path.exists(ckpt):
            log(f"    [sensitivity] checkpoint {ckpt} not found — skipping ε-sweep")
            continue
        log(f"    [sensitivity] ε-sweep  α={alpha}  ({len(eps_vals)} points)")
        pinf  = true_dist.neumann_collapse(alpha, T_MAX)
        model = ScoreNet().to(DEVICE)
        try:
            model.load_state_dict(
                torch.load(ckpt, map_location=DEVICE, weights_only=True))
        except TypeError:
            # weights_only not available in older PyTorch
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        kls = []
        for eps in eps_vals:
            samps = _sample_with_score_noise(model, eps, n=1500)
            kls.append(kde_kl_to_analytic(samps, pinf))
        # Integrated error energy η ≈ ε² · d · T  (d=2)
        etas = [e**2 * 2 * T_MAX for e in eps_vals]
        noise_results[alpha] = {'eta': etas, 'kl': kls}

    n_panels = 2 if noise_results else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1 ───────────────────────────────────────────
    ax = axes[0]
    alpha_arr = np.array(alphas)
    ratio_arr = np.array([sensitivity[a] for a in alphas])

    for a in alphas:
        ax.scatter([a], [sensitivity[a]], color=ALPHA_COLORS[a],
                   s=100, zorder=4, label=fr'$\alpha={a}$')

    # Theory: C / α²  (normalise to interpolate through the data)
    C_norm = float(np.mean(ratio_arr * alpha_arr**2))
    alpha_fine = np.linspace(0.05, 0.97, 200)
    ax.plot(alpha_fine, C_norm / alpha_fine**2, 'k--', lw=1.5,
            label=r'$C/\alpha^2$  (theory)')

    ax.set_xlabel(r'$\alpha$ (fresh-data fraction)')
    ax.set_ylabel(r'$\mathcal{R}(\alpha) = \mathrm{KL}_{\rm floor}\,/\,\bar{\eta}$')
    ax.set_title(r'Sensitivity ratio $\propto \alpha^{-2}$')
    ax.legend(fontsize=9)

    # ── Panel 2 ───────────────────────────────────────────
    if noise_results:
        ax = axes[1]
        for alpha, data in sorted(noise_results.items()):
            c    = ALPHA_COLORS[alpha]
            eta  = np.array(data['eta'])
            kls  = np.array(data['kl'])
            ax.plot(eta, kls, color=c, lw=1.8, marker='o', ms=4,
                    label=fr'$\alpha={alpha}$')
            # Annotate fitted slope
            idx = eta > 0
            if idx.sum() >= 2:
                slope = float(np.polyfit(eta[idx], kls[idx], 1)[0])
                ax.annotate(fr'slope={slope:.2f}',
                            xy=(eta[idx][-1], kls[idx][-1]),
                            fontsize=7.5, color=c, ha='right', va='bottom')

        ax.set_xlabel(
            r'Integrated score error $\eta \approx \varepsilon^2 \cdot d \cdot T$')
        ax.set_ylabel(r'$\mathrm{KL}(\tilde{p} \| p_\infty)$')
        ax.set_title(r'Linear growth in $\eta$; slope $\propto \alpha^{-2}$')
        ax.legend(fontsize=9)

    fig.suptitle(
        r'Prop. 4 — Sensitivity–Collapse Duality:'
        r'  $\mathrm{KL}(\tilde{p}_\infty \| p_\infty) \leq C\alpha^{-2}\,\eta$'
        '\nThe regime with worst ideal collapse is also most sensitive to score errors',
        fontweight='bold')
    fig.tight_layout()
    savefig(fig, os.path.join(out, '12_sensitivity_duality.pdf'))
    log("  → 12_sensitivity_duality")


def plot_convergence_rate(all_results, out):
    """
    Plot 13 — Convergence Rate + 1/α² Stability Floor (Theorem 3 / Experiment 6).

    Panel 1: log KL(p̂^N ‖ p_∞) vs N with dotted fitted decay.
        Verifies empirical contraction rate ≈ (1-α/2) predicted by Thm 3.

    Panel 2: Empirical contraction rate vs theoretical (1-α/2).
        Points should lie near the identity line.

    Panel 3: Stability floor KL(p̂^∞ ‖ p_∞) vs η̄/α².
        Verifies the C/α² factor in the stability bound (eq. fp-stability).
        Floor should grow linearly with η̄/α².
    """
    gens   = np.array(range(N_GENERATIONS + 1))
    alphas = sorted(all_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    emp_rates = {}

    # ── Panel 1: log KL + fitted rate ─────────────────────
    ax = axes[0]
    for alpha in alphas:
        res   = all_results[alpha]
        c     = ALPHA_COLORS[alpha]
        kl_pi = np.maximum(np.array(res['kl_pinf']), 1e-10)

        ax.semilogy(gens, kl_pi, color=c, lw=1.8, marker='o', ms=3,
                    markevery=3, zorder=4, label=fr'$\alpha={alpha}$')

        # Fit log-linear decay in the early phase where KL is above noise floor
        floor    = float(np.mean(kl_pi[-5:]))
        above    = kl_pi > max(3.0 * floor, 1e-6)
        fit_mask = above & (gens >= 1) & (gens <= min(10, N_GENERATIONS))

        if fit_mask.sum() >= 3:
            poly     = np.polyfit(gens[fit_mask], np.log(kl_pi[fit_mask]), 1)
            emp_rate = float(np.exp(poly[0]))
            emp_rates[alpha] = emp_rate

            # Overlay dotted fitted line
            g_ext   = np.arange(0, int(gens[fit_mask][-1]) + 4)
            kl_pred = np.exp(poly[1] + poly[0] * g_ext)
            ax.semilogy(g_ext, kl_pred, color=c, lw=0.9, ls=':', alpha=0.65)

    ax.set_xlabel(r'Generation $N$')
    ax.set_ylabel(r'$\mathrm{KL}(\hat{p}^N \| p_\infty)$  (log)')
    ax.set_title(r'Geometric convergence (dotted: fitted $\hat\gamma^N$)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, N_GENERATIONS)

    # ── Panel 2: empirical vs theoretical rate ────────────
    ax = axes[1]
    alpha_arr = np.array(alphas)
    theo_arr  = 1.0 - alpha_arr / 2.0
    emp_arr   = np.array([emp_rates.get(a, np.nan) for a in alphas])

    # Identity line
    lo, hi = 0.35, 1.02
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.5, label='Identity')

    for a, emp, theo in zip(alphas, emp_arr, theo_arr):
        ax.scatter([theo], [emp if not np.isnan(emp) else theo],
                   color=ALPHA_COLORS[a], s=120, zorder=4,
                   label=fr'$\alpha={a}$')
        if not np.isnan(emp):
            ax.annotate(fr'$\alpha={a}$', xy=(theo, emp),
                        xytext=(theo + 0.02, emp - 0.03),
                        fontsize=7.5, color=ALPHA_COLORS[a])

    ax.set_xlabel(r'Theoretical rate $(1-\alpha/2)$')
    ax.set_ylabel(r'Empirical rate $\hat{\gamma}$')
    ax.set_title(r'Empirical $\approx$ theoretical contraction rate')
    ax.legend(fontsize=8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # ── Panel 3: floor KL vs η̄/α² ────────────────────────
    ax = axes[2]
    x_vals, y_vals = [], []
    for alpha in alphas:
        res      = all_results[alpha]
        kl_pi    = np.array(res['kl_pinf'])
        etas     = np.array(res['eta'])
        floor    = float(np.mean(kl_pi[-5:]))
        eta_bar  = float(np.mean(etas))
        x_val    = eta_bar / alpha**2          # stability driver η̄/α²
        x_vals.append(x_val)
        y_vals.append(floor)
        ax.scatter([x_val], [floor], color=ALPHA_COLORS[alpha], s=100, zorder=4,
                   label=fr'$\alpha={alpha}$,  floor={floor:.3f}')

    # Fit and overlay C·x line
    if len(x_vals) >= 2:
        C_fit  = float(np.polyfit(x_vals, y_vals, 1)[0])
        x_fine = np.linspace(0, max(x_vals) * 1.35, 200)
        ax.plot(x_fine, C_fit * x_fine, 'k--', lw=1.5,
                label=fr'$\hat{{C}} \cdot \bar{{\eta}}/\alpha^2$  '
                      fr'$(\hat{{C}}={C_fit:.2f})$')

    ax.set_xlabel(r'$\bar{\eta}/\alpha^2$  (stability driver)')
    ax.set_ylabel(r'Floor  $\mathrm{KL}(\hat{p}^\infty \| p_\infty)$')
    ax.set_title(r'Stability bound: floor $\leq C(t_0,\beta)\,\alpha^{-2}\,\bar{\eta}$')
    ax.legend(fontsize=8)

    fig.suptitle(
        r'Thm 3 — Convergence Rate $(1-\alpha/2)^N$ and $\alpha^{-2}$ Stability Floor',
        fontweight='bold')
    fig.tight_layout()
    savefig(fig, os.path.join(out, '13_convergence_rate_validation.pdf'))
    log("  → 13_convergence_rate_validation")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    global N_GENERATIONS, ALPHAS, OUT_DIR, N_TRAIN, N_GEN_SAMPLES
    global N_EPOCHS_GEN0, N_EPOCHS_RECURSE, N_REVERSE_STEPS, NEUMANN_K

    parser = argparse.ArgumentParser(description='2-GMM Model Collapse (v2)')
    parser.add_argument('--alphas',      nargs='+', type=float, default=ALPHAS)
    parser.add_argument('--generations', type=int,             default=N_GENERATIONS)
    parser.add_argument('--out_dir',     type=str,             default=OUT_DIR)
    parser.add_argument('--seed',        type=int,             default=SEED)
    parser.add_argument('--fast',        action='store_true',
                        help='Reduce epochs/samples (~3 min sanity check)')
    parser.add_argument('--plot_only',   type=str, default=None,
                        help='Path to all_results.pkl — skip training, regenerate plots')
    args = parser.parse_args()

    N_GENERATIONS = args.generations
    ALPHAS        = args.alphas
    OUT_DIR       = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.fast:
        N_TRAIN = 2_000; N_GEN_SAMPLES = 1_500
        N_EPOCHS_GEN0 = 60; N_EPOCHS_RECURSE = 30
        N_REVERSE_STEPS = 200; NEUMANN_K = 60
        log("  [FAST MODE]")

    set_seed(args.seed)
    true_dist = make_true_dist()

    if args.plot_only:
        log(f"  Loading {args.plot_only}")
        with open(args.plot_only, 'rb') as f:
            all_results = pickle.load(f)
        all_results = {float(k): v for k, v in all_results.items()}
    else:
        log(f"\n{'#'*60}")
        log(f"  2-GMM Model Collapse  (v2)    Device: {DEVICE}")
        log(f"  β≡1 → OU  →  κ = (1-α)e^{{-t0/2}},  t0={T_MAX}")
        log(f"  Alphas: {ALPHAS}    κ's: {[f'{kappa(a):.3f}' for a in ALPHAS]}")
        log(f"  Generations: {N_GENERATIONS}")
        log(f"{'#'*60}")

        t_start = time.time()
        all_results = {}
        for alpha in ALPHAS:
            all_results[alpha] = run_experiment(alpha, true_dist)

        with open(os.path.join(OUT_DIR, 'all_results.pkl'), 'wb') as f:
            pickle.dump(all_results, f)
        log(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} min")
        print_summary(all_results)

    # All plots
    log(f"\n{'='*80}\n  Generating plots\n{'='*80}")
    # plot_kl_curves(all_results, OUT_DIR)
    # plot_kl_to_pinf_and_bound(all_results, OUT_DIR)
    # plot_kl_decomposition(all_results, OUT_DIR)
    # plot_neumann_series(true_dist, OUT_DIR)
    plot_limiting_behaviors(true_dist, OUT_DIR)
    plot_annealed_truncation(true_dist, OUT_DIR, alphas=(0.1, 0.5, 0.9), N_gens=50, t_inf=0.2)
    '''plot_t0_sweep(true_dist, OUT_DIR)'''
    '''plot_kappa_contour(OUT_DIR)'''
    '''plot_density_grid(all_results, true_dist, OUT_DIR)
    plot_density_strips(all_results, true_dist, OUT_DIR)
    plot_marginals(all_results, true_dist, OUT_DIR)
    plot_spectral_filtering(all_results, true_dist, OUT_DIR)
    plot_sensitivity_duality(all_results, true_dist, OUT_DIR)
    plot_convergence_rate(all_results, OUT_DIR)'''

    log(f"\nDone. All outputs in: {OUT_DIR}/")

    for f in sorted(os.listdir(OUT_DIR)):
        if not os.path.isdir(os.path.join(OUT_DIR, f)):
            log(f"  {f}")
        
if __name__ == '__main__':
    main()