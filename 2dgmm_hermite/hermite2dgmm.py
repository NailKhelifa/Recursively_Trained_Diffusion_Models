#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import copy
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal, gaussian_kde
from scipy.linalg import sqrtm
from numpy.polynomial.hermite_e import hermeval

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],               # <--- Forces standard text to Computer Modern
    "mathtext.fontset": "cm",              # <--- Forces math text to Computer Modern
    "axes.formatter.use_mathtext": True,   # <--- Fixes minus signs on tick labels
    "font.size": 16, 
    "axes.labelsize": 16, 
    "axes.titlesize": 16,
    "legend.fontsize": 14, 
    "xtick.labelsize": 14, 
    "ytick.labelsize": 14,
    "figure.dpi": 200, 
    "axes.linewidth": 0.8, 
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "grid.alpha"    : 0.3,
    "grid.linewidth": 0.5,
    "axes.grid"      : False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.spines.top": True,   
    "axes.spines.right": True, 
})

COLORS = {
    "blue": "#2660A4",
    "red": "#E63946",
    "green": "#2A9D8F",
    "orange": "#E76F51",
    "purple": "#6C5B7B",
    "gray": "#8D99AE",
    "dark": "#264653",
    "gold": "#E9C46A",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def savefig(fig, outdir, name):
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {name}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------------
# 2D GMM
# ---------------------------------------------------------------------
class GaussianMixture2D:
    def __init__(self, weights, means, covs):
        self.weights = np.asarray(weights, dtype=float)
        self.weights /= self.weights.sum()
        self.means = [np.asarray(m, dtype=float) for m in means]
        self.covs = [np.asarray(c, dtype=float) for c in covs]
        self.K = len(self.weights)

    def sample(self, n, rng):
        comp = rng.choice(self.K, size=n, p=self.weights)
        out = np.zeros((n, 2), dtype=np.float32)
        for k in range(self.K):
            idx = np.where(comp == k)[0]
            if len(idx):
                out[idx] = rng.multivariate_normal(
                    self.means[k], self.covs[k], size=len(idx)
                ).astype(np.float32)
        return out

    def pdf(self, x):
        out = np.zeros(x.shape[:-1], dtype=float)
        for w, m, c in zip(self.weights, self.means, self.covs):
            out += w * multivariate_normal.pdf(x, mean=m, cov=c)
        return out

# ---------------------------------------------------------------------
# OU forward process
#   X_t = e^{-t/2} X_0 + sqrt(1-e^{-t}) eps
#   eps ~ N(0,I)
# ---------------------------------------------------------------------
def ou_mean_coeff(t):
    return torch.exp(-t / 2)

def ou_var(t):
    return 1.0 - torch.exp(-t)

def ou_std(t):
    return torch.sqrt(torch.clamp(ou_var(t), min=1e-8))

def forward_sample(x0, t):
    eps = torch.randn_like(x0)
    xt = ou_mean_coeff(t)[:, None] * x0 + ou_std(t)[:, None] * eps
    return xt, eps

# ---------------------------------------------------------------------
# Time embedding + score net
# Predict epsilon, like in your CIFAR script
# ---------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ScoreNet2D(nn.Module):
    def __init__(self, hidden=128, tdim=64):
        super().__init__()
        self.temb = nn.Sequential(
            SinusoidalPosEmb(tdim),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.net = nn.Sequential(
            nn.Linear(2 + tdim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        te = self.temb(t)
        h = torch.cat([x, te], dim=-1)
        return self.net(h)

# ---------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

# ---------------------------------------------------------------------
# Train score model on samples from q_i
# ---------------------------------------------------------------------
def train_score_network_2d(
    model,
    data_tensor,
    device="cpu",
    batch_size=512,
    n_epochs=400,
    lr=2e-3,
    ema_decay=0.999,
    t0=0.25,
    T=3.0,
):
    model = model.to(device)
    loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size,
                        shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema = EMA(model, decay=ema_decay)

    losses = []
    model.train()
    for ep in range(n_epochs):
        total = 0.0
        nb = 0
        for (x0,) in loader:
            x0 = x0.to(device)
            B = x0.shape[0]
            t = t0 + (T - t0) * torch.rand(B, device=device)
            xt, eps = forward_sample(x0, t)
            eps_pred = model(xt, t)
            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            total += loss.item()
            nb += 1

        losses.append(total / max(nb, 1))
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"      epoch {ep+1:4d}/{n_epochs}   loss={losses[-1]:.6f}")

    return ema, losses

# ---------------------------------------------------------------------
# Smooth Hermite-perturbed Gaussian
# p(x) \propto phi(x) * g(x)^2
# where phi = N(0, I_2)
# ---------------------------------------------------------------------
class HermitePerturbedGaussian2D:
    def __init__(self, coeffs):
        """
        coeffs: dict with keys (n1, n2) and scalar values
                defining g(x) = 1 + sum a_{n1,n2} h_{n1,n2}(x)
        where h_{n1,n2} uses orthonormal probabilists' Hermites.
        """
        self.coeffs = coeffs

    def h1d(self, n, x):
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0
        return hermeval(x, coeffs) / math.sqrt(math.factorial(n))

    def g(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        out = np.ones(len(x), dtype=np.float64)
        for (n1, n2), a in self.coeffs.items():
            out += a * self.h1d(n1, x1) * self.h1d(n2, x2)
        return out

    def sample(self, n, rng, proposal_mult=20):
        """
        Importance-resampling from standard Gaussian:
        target density proportional to phi(x) * g(x)^2
        """
        m = proposal_mult * n
        z = rng.normal(size=(m, 2)).astype(np.float32)
        w = self.g(z) ** 2
        w = np.maximum(w, 1e-12)
        w = w / w.sum()
        idx = rng.choice(m, size=n, replace=True, p=w)
        return z[idx].astype(np.float32)

    def unnormalized_pdf(self, x):
        phi = multivariate_normal.pdf(x, mean=np.zeros(2), cov=np.eye(2))
        return phi * (self.g(x.reshape(-1, 2)) ** 2)
    
def hermite_coeff_matrix_fixed_gaussian(samples, max_degree=6):
    z = np.asarray(samples)
    z1, z2 = z[:, 0], z[:, 1]

    H1 = [hermite_1d(n, z1) / math.sqrt(math.factorial(n)) for n in range(max_degree + 1)]
    H2 = [hermite_1d(n, z2) / math.sqrt(math.factorial(n)) for n in range(max_degree + 1)]

    C = np.zeros((max_degree + 1, max_degree + 1))
    for n1 in range(max_degree + 1):
        for n2 in range(max_degree + 1):
            C[n1, n2] = np.mean(H1[n1] * H2[n2])
    return C
# ---------------------------------------------------------------------
# Reverse SDE sampler
# dY_s = [-1/2 Y_s - score(Y_s,s)] ds + dBbar_s
# with score = -eps_pred / sigma_t
# Integrate backward from T to t0
# ---------------------------------------------------------------------
@torch.no_grad()
def sample_reverse_sde_2d(
    model,
    n_samples,
    device="cpu",
    t0=0.25,
    T=3.0,
    n_steps=500,
    batch_size=2000,
):
    model.eval()
    out = []
    remaining = n_samples

    while remaining > 0:
        bs = min(batch_size, remaining)
        x = torch.randn(bs, 2, device=device)
        ts = torch.linspace(T, t0, n_steps + 1, device=device)
        dt = (T - t0) / n_steps

        for i in range(n_steps):
            tcur = ts[i]
            tvec = tcur.expand(bs)
            sigma = ou_std(tvec)[:, None]
            eps_pred = model(x, tvec)
            score = -eps_pred / sigma
            drift = -0.5 * x - score
            noise = torch.randn_like(x) if i < n_steps - 1 else 0.0
            x = x - drift * dt + math.sqrt(dt) * noise

        out.append(x.cpu())
        remaining -= bs

    return torch.cat(out, dim=0)

# ---------------------------------------------------------------------
# Exact reference samplers
# ---------------------------------------------------------------------
def sample_exact_errorfree_iterates(gmm, alpha, t0, n_samples, n_generations, rng):
    p0 = gmm.sample(n_samples, rng)
    out = [p0.copy()]
    current = p0.copy()

    for _ in range(n_generations):
        mask = rng.uniform(size=n_samples) < alpha
        fresh = gmm.sample(mask.sum(), rng)
        mix = np.zeros((n_samples, 2), dtype=np.float32)
        mix[mask] = fresh
        mix[~mask] = current[~mask]

        a = math.exp(-t0 / 2.0)
        b = math.sqrt(1.0 - math.exp(-t0))
        z = rng.normal(size=mix.shape).astype(np.float32)
        current = a * mix + b * z
        out.append(current.copy())
    return out

def sample_neumann_limit(gmm, alpha, t0, n_samples, rng):
    k = rng.geometric(p=alpha, size=n_samples) - 1
    x0 = gmm.sample(n_samples, rng)
    tt = (k + 1) * t0
    a = np.exp(-tt / 2.0)[:, None].astype(np.float32)
    b = np.sqrt(1.0 - np.exp(-tt))[:, None].astype(np.float32)
    z = rng.normal(size=x0.shape).astype(np.float32)
    return a * x0 + b * z

def sample_neumann_limit_dist(dist, alpha, t0, n_samples, rng):
    """
    Sample from p_inf = alpha * sum_{k>=0} (1-alpha)^k U_{(k+1)t0}(p_data)
    for any dist object with method sample(n, rng).
    """
    k = rng.geometric(p=alpha, size=n_samples) - 1
    x0 = dist.sample(n_samples, rng)

    tt = (k + 1) * t0
    a = np.exp(-tt / 2.0)[:, None].astype(np.float32)
    b = np.sqrt(1.0 - np.exp(-tt))[:, None].astype(np.float32)

    z = rng.normal(size=x0.shape).astype(np.float32)
    return a * x0 + b * z

def sample_exact_errorfree_iterates_dist(dist, alpha, t0, n_samples, n_generations, rng):
    p0 = dist.sample(n_samples, rng)
    out = [p0.copy()]
    current = p0.copy()

    for _ in range(n_generations):
        mask = rng.uniform(size=n_samples) < alpha
        fresh = dist.sample(mask.sum(), rng)

        mix = np.zeros((n_samples, 2), dtype=np.float32)
        mix[mask] = fresh
        mix[~mask] = current[~mask]

        a = math.exp(-t0 / 2.0)
        b = math.sqrt(1.0 - math.exp(-t0))
        z = rng.normal(size=mix.shape).astype(np.float32)
        current = a * mix + b * z
        out.append(current.copy())

    return out

# ---------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------
def mean_cov(x):
    x = np.asarray(x)
    return x.mean(axis=0), np.cov(x.T)

def gaussian_w2(mu1, cov1, mu2, cov2):
    s1 = sqrtm(cov1)
    if np.iscomplexobj(s1):
        s1 = s1.real
    mid = sqrtm(s1 @ cov2 @ s1)
    if np.iscomplexobj(mid):
        mid = mid.real
    val = (mu1 - mu2) @ (mu1 - mu2) + np.trace(cov1 + cov2 - 2 * mid)
    return float(np.sqrt(max(val, 0.0)))

def pairwise_sq_dists(x, y):
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return x2 + y2 - 2 * x @ y.T

def median_heuristic(x, y, max_points=2000):
    z = np.vstack([x[:max_points], y[:max_points]])
    d2 = pairwise_sq_dists(z, z)
    tri = d2[np.triu_indices_from(d2, k=1)]
    tri = tri[tri > 0]
    return max(float(np.median(tri)), 1e-6)

def mmd_rbf(x, y, sigma2=None):
    if sigma2 is None:
        sigma2 = median_heuristic(x, y)
    kxx = np.exp(-pairwise_sq_dists(x, x) / (2 * sigma2))
    kyy = np.exp(-pairwise_sq_dists(y, y) / (2 * sigma2))
    kxy = np.exp(-pairwise_sq_dists(x, y) / (2 * sigma2))
    n = len(x)
    m = len(y)
    val = (
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2 * kxy.mean()
    )
    return float(max(val, 0.0))

def sliced_w2(x, y, n_proj=128, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    d = x.shape[1]
    vals = []
    for _ in range(n_proj):
        theta = rng.normal(size=d)
        theta /= np.linalg.norm(theta) + 1e-12
        xp = np.sort(x @ theta)
        yp = np.sort(y @ theta)
        m = min(len(xp), len(yp))
        vals.append(np.mean((xp[:m] - yp[:m]) ** 2))
    return float(np.sqrt(np.mean(vals)))
# ---------------------------------------------------------------------
# Hermite coefficients in 2D
# ---------------------------------------------------------------------
def hermite_1d(n, x):
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return hermeval(x, coeffs)

def standardize(samples, ref):
    mu, cov = mean_cov(ref)
    vals, vecs = np.linalg.eigh(cov)
    W = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-10))) @ vecs.T
    z = (samples - mu) @ W.T
    return z

def hermite_coeff_matrix(samples, ref_samples, max_degree=6):
    z = standardize(samples, ref_samples)
    z1, z2 = z[:, 0], z[:, 1]
    
    # Crucial fix: divide by sqrt(n!) to make the basis orthonormal
    H1 = [hermite_1d(n, z1) / math.sqrt(math.factorial(n)) for n in range(max_degree + 1)]
    H2 = [hermite_1d(n, z2) / math.sqrt(math.factorial(n)) for n in range(max_degree + 1)]

    C = np.zeros((max_degree + 1, max_degree + 1))
    for n1 in range(max_degree + 1):
        for n2 in range(max_degree + 1):
            C[n1, n2] = np.mean(H1[n1] * H2[n2])
    return C

def degree_ratio(C, Cref, eps=1e-12):
    E = degree_energy(C)
    Eref = degree_energy(Cref)
    return E / np.maximum(Eref, eps)

def degree_error(C, Ctarget):
    K = C.shape[0] - 1
    out = []
    for k in range(2 * K + 1):
        s = 0.0
        for n1 in range(K + 1):
            for n2 in range(K + 1):
                if n1 + n2 == k:
                    s += (C[n1, n2] - Ctarget[n1, n2]) ** 2
        out.append(s)
    return np.array(out)

def degree_energy(C):
    K = C.shape[0] - 1
    out = []
    for k in range(2 * K + 1):
        s = 0.0
        for n1 in range(K + 1):
            for n2 in range(K + 1):
                if n1 + n2 == k:
                    s += C[n1, n2] ** 2
        out.append(s)
    return np.array(out)

def triangular_mask(K):
    mask = np.zeros((K + 1, K + 1), dtype=bool)
    for n1 in range(K + 1):
        for n2 in range(K + 1):
            if n1 + n2 > K:
                mask[n1, n2] = True
    return mask


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def kde_contour(ax, samples, color, levels=8, label=None):
    kde = gaussian_kde(samples.T)
    xs = np.linspace(-6.0, 6.0, 220)
    ys = np.linspace(-5.0, 7.0, 220)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X, Y], axis=-1)
    Z = kde(grid.reshape(-1, 2).T).reshape(X.shape)
    ax.contour(X, Y, Z, levels=levels, colors=[color], linewidths=1.4)
    if label is not None:
        ax.plot([], [], color=color, label=label)

# ---------------------------------------------------------------------
# Fourier
# ---------------------------------------------------------------------

def density_on_grid(samples, xlim=(-7, 7), ylim=(-4, 4), bins=256):
    H, xedges, yedges = np.histogram2d(
        samples[:, 0], samples[:, 1],
        bins=bins,
        range=[xlim, ylim],
        density=True
    )
    # histogram2d returns shape [xbins, ybins], transpose for plotting convention
    return H.T, xedges, yedges

def fft_power_from_density(H):
    F = np.fft.fftshift(np.fft.fft2(H))
    P = np.abs(F) ** 2
    return P

def log_power(P, eps=1e-12):
    return np.log10(P + eps)

# ============================================================
# High-frequency Hermite probe distribution + clean plots
# ============================================================

import os
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ============================================================
# 1. Oscillatory Gaussian density
# ============================================================

class OscillatoryGaussian2D:
    """
    p_data(x) ∝ gamma(x) * exp(phi(x)),
    where gamma = N(0, I_2).

    The oscillatory bounded tilt creates high Hermite modes while
    remaining easy to sample by importance resampling from gamma.
    """

    def __init__(
        self,
        omega=3.7,
        beta1=1.25,
        beta2=0.85,
        beta3=0.45,
        proposal_mult=80,
    ):
        self.omega = float(omega)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.beta3 = float(beta3)
        self.proposal_mult = int(proposal_mult)

    def log_weight(self, x):
        """
        x: [N,2]
        returns log unnormalized density ratio relative to N(0,I).
        """
        x = np.asarray(x, dtype=np.float64)
        x1 = x[:, 0]
        x2 = x[:, 1]

        w = self.omega

        phi = (
            self.beta1 * np.cos(w * (x1 + 0.7 * x2))
            + self.beta2 * np.cos(w * x1) * np.cos((w - 1.0) * x2)
            + self.beta3 * np.sin((w + 1.0) * x1 - 0.4 * x2)
        )

        return phi

    def sample(self, n, rng):
        """
        Importance-resampling from standard Gaussian.

        Since exp(phi) is bounded, this is stable. Increase proposal_mult
        if you want lower resampling noise.
        """
        m = self.proposal_mult * n
        z = rng.normal(size=(m, 2)).astype(np.float32)

        lw = self.log_weight(z)
        lw = lw - lw.max()
        w = np.exp(lw)
        w = w / w.sum()

        idx = rng.choice(m, size=n, replace=True, p=w)
        return z[idx].astype(np.float32)

    def weight_on_grid(self, X, Y):
        xy = np.stack([X.ravel(), Y.ravel()], axis=-1)
        lw = self.log_weight(xy)
        return np.exp(lw).reshape(X.shape)


# ============================================================
# 2. Exact Hermite attenuation predicted by the theorem
# ============================================================

def hermite_multiplier_matrix(alpha, t0, max_degree):
    """
    M[n1,n2] = m_{n1+n2}(alpha,t0)
             = alpha exp(-(n1+n2)t0/2)
               / (1 - (1-alpha) exp(-(n1+n2)t0/2)).
    """
    M = np.zeros((max_degree + 1, max_degree + 1))

    for n1 in range(max_degree + 1):
        for n2 in range(max_degree + 1):
            k = n1 + n2
            e = np.exp(-0.5 * k * t0)
            M[n1, n2] = alpha * e / (1.0 - (1.0 - alpha) * e)

    return M


def theoretical_limit_coeff_matrix(C_data, alpha, t0):
    """
    Exact Hermite coefficients of p_inf^star predicted by Proposition spectral.
    """
    K = C_data.shape[0] - 1
    M = hermite_multiplier_matrix(alpha, t0, K)
    return M * C_data


def nonconstant_vmax(*Cs, q=99.5):
    """
    Common symmetric color scale excluding the constant coefficient C_00.
    """
    vals = []
    for C in Cs:
        D = C.copy()
        D[0, 0] = 0.0
        vals.append(np.abs(D).ravel())

    vals = np.concatenate(vals)
    return float(np.percentile(vals, q))


def prepare_coeff_for_plot(C, remove_constant=True):
    D = C.copy()
    if remove_constant:
        D[0, 0] = 0.0
    return D


# ============================================================
# 3. Clean Hermite heatmap plot
# ============================================================

def plot_hermite_heatmaps_clean(
    alpha_dict,
    p_data_eval,
    alphas,
    t0,
    max_degree,
    outdir,
    n_generations,
    use_theoretical_limit=True,
    name="hermite_heatmaps_clean_highfreq",
):
    """
    Plots p_data, theoretical/sampled p_inf, and learned final iterate.

    Recommended:
        use_theoretical_limit=True

    This makes the middle column exactly the theorem prediction:
        C_inf[n1,n2] = m_{n1+n2} C_data[n1,n2].
    """
    C_data = hermite_coeff_matrix_fixed_gaussian(
        p_data_eval,
        max_degree=max_degree,
    )

    n_rows = len(alphas)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(10.5, 3.4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for i, alpha in enumerate(alphas):
        data = alpha_dict[alpha]

        if use_theoretical_limit:
            C_inf = theoretical_limit_coeff_matrix(C_data, alpha, t0)
        else:
            C_inf = hermite_coeff_matrix_fixed_gaussian(
                data["p_inf"],
                max_degree=max_degree,
            )

        C_last = hermite_coeff_matrix_fixed_gaussian(
            data["learned_iterates"][-1],
            max_degree=max_degree,
        )

        vmax = nonconstant_vmax(C_data, C_inf, C_last, q=99.5)

        matrices = [
            prepare_coeff_for_plot(C_data),
            prepare_coeff_for_plot(C_inf),
            prepare_coeff_for_plot(C_last),
        ]

        titles = [
            r"$p_{\mathrm{data}}$",
            r"$p_\infty^\star$ (theory)",
            rf"learned $\hat p^{{{n_generations}}}$",
        ]

        for j in range(3):
            ax = axes[i, j]
            im = ax.imshow(
                matrices[j],
                origin="lower",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )

            if i == 0:
                ax.set_title(titles[j], pad=8)

            if j == 0:
                ax.set_ylabel(fr"${alpha:.1f}$" + "\n" + r"$n_1$")
            else:
                ax.set_ylabel("")

            if i == n_rows - 1:
                ax.set_xlabel(r"$n_2$")

            # Clean ticks
            ticks = [0, 5, 10, 15, 20]
            ticks = [t for t in ticks if t <= max_degree]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(length=0)

            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_linewidth(0.8)

        cbar = fig.colorbar(
            im,
            ax=axes[i, :],
            location="right",
            fraction=0.035,
            pad=0.02,
        )
        cbar.set_label(r"Hermite coefficient", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    # fig.suptitle(
    #     r"Hermite-mode attenuation under recursive smoothing",
    #     fontsize=22,
    #     y=1.01,
    # )

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 4. Empirical W2 / sliced W2 utilities
# ============================================================

def empirical_w2_subsample(x, y, n_ot=1200, seed=0):
    """
    Exact empirical W2 between two equal-size subsamples using Hungarian matching.

    This is O(n^3), so keep n_ot around 800--1500.
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = min(len(x), len(y), n_ot)
    ix = rng.choice(len(x), size=n, replace=False)
    iy = rng.choice(len(y), size=n, replace=False)

    xs = x[ix]
    ys = y[iy]

    C = cdist(xs, ys, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(C)

    return float(np.sqrt(C[row_ind, col_ind].mean()))


def sliced_w2_fast(x, y, n_proj=512, seed=0):
    """
    Fast sliced-W2 proxy. More stable for large samples.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    vals = []
    for _ in range(n_proj):
        theta = rng.normal(size=2)
        theta = theta / (np.linalg.norm(theta) + 1e-12)

        xp = np.sort(x @ theta)
        yp = np.sort(y @ theta)

        vals.append(np.mean((xp - yp) ** 2))

    return float(np.sqrt(np.mean(vals)))


# ============================================================
# 5. Plot W2(\hat p_i, p_inf^star)
# ============================================================

def plot_distance_to_limit(
    alpha_dict,
    alphas,
    n_generations,
    outdir,
    use_exact_w2=True,
    n_ot=1000,
    name="w2_to_limit_clean",
):
    """
    Plots distance to p_inf^star for both exact error-free recursion and learned recursion.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    colors = plt.cm.cividis(np.linspace(0.12, 0.85, len(alphas)))
    N = np.arange(n_generations + 1)

    for idx, (alpha, color) in enumerate(zip(alphas, colors)):
        data = alpha_dict[alpha]
        p_inf = data["p_inf"]

        d_exact = []
        d_learned = []

        for g in range(n_generations + 1):
            x_exact = data["exact_iterates"][g]
            x_learn = data["learned_iterates"][g]

            if use_exact_w2:
                d_exact.append(
                    empirical_w2_subsample(
                        x_exact,
                        p_inf,
                        n_ot=n_ot,
                        seed=1000 + 17 * g + idx,
                    )
                )
                d_learned.append(
                    empirical_w2_subsample(
                        x_learn,
                        p_inf,
                        n_ot=n_ot,
                        seed=2000 + 17 * g + idx,
                    )
                )
            else:
                d_exact.append(
                    sliced_w2_fast(
                        x_exact,
                        p_inf,
                        n_proj=512,
                        seed=1000 + 17 * g + idx,
                    )
                )
                d_learned.append(
                    sliced_w2_fast(
                        x_learn,
                        p_inf,
                        n_proj=512,
                        seed=2000 + 17 * g + idx,
                    )
                )

        ax.plot(
            N,
            d_exact,
            "--",
            color=color,
            lw=2.2,
            alpha=0.75,
            label=fr"$\alpha={alpha}$ exact",
        )

        ax.plot(
            N,
            d_learned,
            "-o",
            color=color,
            lw=2.2,
            ms=4,
            mec="white",
            mew=0.5,
            label=fr"$\alpha={alpha}$ learned",
        )

    ylabel = (
        r"empirical $W_2(\hat p^N,p_\infty^\star)$"
        if use_exact_w2
        else r"sliced $W_2(\hat p^N,p_\infty^\star)$"
    )

    ax.set_xlabel(r"recursive generation $N$")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=10, ncols=2)
    ax.set_title(r"Convergence toward the collapse distribution")

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 6. Limiting behavior plots
# ============================================================

def ou_push_samples(x, t, rng):
    """
    Apply U_t to existing samples.
    """
    x = np.asarray(x, dtype=np.float32)
    a = math.exp(-0.5 * t)
    b = math.sqrt(1.0 - math.exp(-t))
    z = rng.normal(size=x.shape).astype(np.float32)
    return a * x + b * z


def plot_limiting_behaviors(
    dist,
    outdir,
    seed=123,
    n_samples=20000,
    n_ot=1000,
    alpha_fixed=0.5,
    t0_fixed=0.10,
    name="limiting_behaviors_w2",
):
    """
    Empirical illustration of Theorem limiting-behavior statements:

    (i)   alpha -> 1: p_inf -> U_{t0}(p_data)
    (ii)  alpha -> 0: p_inf -> N(0,I)
    (iii) t0 -> 0:   p_inf -> p_data
    (iv)  t0 > 0:    p_inf != p_data, unless p_data is Gaussian
    """
    rng = np.random.default_rng(seed)

    # Shared reference samples
    pdata = dist.sample(n_samples, rng)
    normal = rng.normal(size=(n_samples, 2)).astype(np.float32)
    ut0_pdata = ou_push_samples(pdata, t0_fixed, rng)

    # --------------------------------------------------------
    # alpha limits
    # --------------------------------------------------------
    alpha_grid = np.array(
        [0.02, 0.04, 0.07, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.92, 0.97, 0.99]
    )

    d_alpha_to_ut0 = []
    d_alpha_to_gaussian = []
    d_alpha_to_data = []

    for j, alpha in enumerate(alpha_grid):
        pinf = sample_neumann_limit_dist(
            dist,
            alpha=alpha,
            t0=t0_fixed,
            n_samples=n_samples,
            rng=rng,
        )

        d_alpha_to_ut0.append(
            empirical_w2_subsample(
                pinf,
                ut0_pdata,
                n_ot=n_ot,
                seed=10_000 + j,
            )
        )

        d_alpha_to_gaussian.append(
            empirical_w2_subsample(
                pinf,
                normal,
                n_ot=n_ot,
                seed=20_000 + j,
            )
        )

        d_alpha_to_data.append(
            empirical_w2_subsample(
                pinf,
                pdata,
                n_ot=n_ot,
                seed=30_000 + j,
            )
        )

    # --------------------------------------------------------
    # t0 limits
    # --------------------------------------------------------
    t0_grid = np.array(
        [0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.25, 0.40, 0.70]
    )

    d_t0_to_data = []
    d_t0_to_gaussian = []

    for j, t0 in enumerate(t0_grid):
        pinf = sample_neumann_limit_dist(
            dist,
            alpha=alpha_fixed,
            t0=t0,
            n_samples=n_samples,
            rng=rng,
        )

        d_t0_to_data.append(
            empirical_w2_subsample(
                pinf,
                pdata,
                n_ot=n_ot,
                seed=40_000 + j,
            )
        )

        d_t0_to_gaussian.append(
            empirical_w2_subsample(
                pinf,
                normal,
                n_ot=n_ot,
                seed=50_000 + j,
            )
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax = axes[0]
    ax.plot(alpha_grid, d_alpha_to_ut0, "-o", lw=2.2, ms=4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"empirical $W_2$")
    ax.set_title(r"(i) $\alpha\to1$: $p_\infty^\star\to \mathcal{U}_{t_0}(p_{\rm data})$")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(alpha_grid, d_alpha_to_gaussian, "-o", lw=2.2, ms=4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_title(r"(ii) $\alpha\to0$: $p_\infty^\star\to\mathcal{N}(0,I)$")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(t0_grid, d_t0_to_data, "-o", lw=2.2, ms=4, label=r"to $p_{\rm data}$")
    ax.plot(t0_grid, d_t0_to_gaussian, "--s", lw=2.0, ms=4, label=r"to $\mathcal{N}(0,I)$")
    ax.set_xlabel(r"$t_0$")
    ax.set_title(r"(iii)--(iv) effect of truncation time")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11)

    fig.suptitle(
        r"Empirical illustration of the limiting behavior of $p_\infty^\star$",
        fontsize=18,
        y=1.03,
    )

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# One-row Hermite heatmap:
# p_data + theoretical p_inf^star for several alphas
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_hermite_pdata_and_theory_limits_row(
    p_data_eval,
    alphas,
    t0,
    max_degree,
    outdir,
    name="hermite_pdata_and_theory_limits_row",
    remove_constant=True,
    q_vmax=99.5,
):
    """
    One-row Hermite coefficient heatmap.

    Columns:
        1. p_data
        2. p_inf^star(alpha_1)
        3. p_inf^star(alpha_2)
        4. p_inf^star(alpha_3)

    The p_inf^star panels are theoretical Hermite coefficients:
        C_inf[n1,n2] = m_{n1+n2}(alpha,t0) C_data[n1,n2].
    """

    # Hermite coefficients of p_data relative to standard Gaussian
    C_data = hermite_coeff_matrix_fixed_gaussian(
        p_data_eval,
        max_degree=max_degree,
    )

    # Theoretical limit coefficients for each alpha
    C_limits = [
        theoretical_limit_coeff_matrix(C_data, alpha, t0)
        for alpha in alphas
    ]

    matrices = [C_data] + C_limits

    # Remove C_00 visually, otherwise it dominates the color scale.
    matrices_plot = [
        prepare_coeff_for_plot(C, remove_constant=remove_constant)
        for C in matrices
    ]

    # Shared symmetric color scale across all panels
    vmax = nonconstant_vmax(*matrices, q=q_vmax)
    vmin = -vmax

    ncols = 1 + len(alphas)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(3.15 * ncols, 3.35),
        constrained_layout=True,
    )

    if ncols == 1:
        axes = [axes]

    titles = [r"$p_{\mathrm{data}}$"] + [
        rf"$p_\infty^\star$, $\alpha={alpha}$"
        for alpha in alphas
    ]

    for j, ax in enumerate(axes):
        im = ax.imshow(
            matrices_plot[j],
            origin="lower",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )

        ax.set_title(titles[j], pad=7)

        ticks = [0, 5, 10, 15, 20, 25, 30]
        ticks = [t for t in ticks if t <= max_degree]

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.tick_params(length=0)

        ax.set_xlabel(r"$n_2$")

        if j == 0:
            ax.set_ylabel(r"$n_1$")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.8)

    cbar = fig.colorbar(
        im,
        ax=axes,
        location="right",
        fraction=0.035,
        pad=0.02,
    )
    cbar.set_label(r"Hermite coefficient", rotation=90)

    # fig.suptitle(
    #     r"Hermite-mode attenuation in the theoretical collapse distribution",
    #     fontsize=20,
    #     y=1.06,
    # )

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(
        os.path.join(outdir, f"{name}.pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(outdir, f"{name}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

# ---------------------------------------------------------------------
# Main recursive learned experiment
# ---------------------------------------------------------------------
def run_experiment(
    outdir="gmm2d_learned_recursive",
    seed=0,
    alphas=[0.1, 0.5, 0.9],
    t0=0.35,
    T=3.0,
    n_generations=8,
    n_train=30000,
    n_eval=12000,
    n_epochs=350,
    batch_size=512,
    lr=2e-3,
    ema_decay=0.995,
    n_steps_sample=500,
    device="cpu",
    max_hermite_degree=6,
    pinf_only=False,
):
    ensure_dir(outdir)
    set_seed(seed)
    cmap = plt.cm.viridis
    rng = np.random.default_rng(seed)

    # producing anti-diagonal bands (ring/annulus)
    '''r = 3.5
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    weights = [1/8] * 8
    means = [np.array([r * np.cos(a), r * np.sin(a)]) for a in angles]
    covs = [0.35 * np.eye(2) for _ in range(8)]

    gmm = GaussianMixture2D(weights, means, covs)'''

    # asymmetric banana ridge

    # -----------------------------------------------------------------
    # The "Asymmetric Banana Ridge" for a fully dense Hermite spectrum
    # -----------------------------------------------------------------
    '''n_components = 18
    weights = []
    means = []
    covs = []
    
    for i in range(n_components):
        # 1. Unequal weights: Density is heavy on the left, fading to the right
        w = 1.0 + (n_components - i) * 0.25 
        weights.append(w)
        
        # 2. Non-linear curved manifold (Asymmetric parabola)
        x = -3.5 + i * (7.0 / (n_components - 1))
        y = 0.35 * (x + 0.5)**2 - x - 2.0
        means.append(np.array([x, y]))
        
        # 3. Covariances aligned strictly with the curve's tangent
        # This maintains a "sharp" density ridge that forces high-degree Hermite polynomials to activate.
        tangent_x = 1.0
        tangent_y = 0.7 * (x + 0.5) - 1.0
        norm = math.sqrt(tangent_x**2 + tangent_y**2)
        tx, ty = tangent_x / norm, tangent_y / norm
        nx, ny = -ty, tx  # Normal vector
        
        # Stretch the Gaussian along the curve, keep it very narrow across the curve
        var_tangent = 0.6
        var_normal = 0.08
        R = np.array([[tx, nx], [ty, ny]])
        cov = R @ np.diag([var_tangent, var_normal]) @ R.T
        covs.append(cov)

    weights = np.array(weights) / np.sum(weights)
    gmm = GaussianMixture2D(weights, means, covs)'''

    """# -----------------------------------------------------------------
    # The "Smooth Asymmetric Pear" for a clean, non-oscillating spectrum
    # -----------------------------------------------------------------
    weights = [0.60, 0.30, 0.10]
    means = [
        np.array([-1.0, -1.0]),  # Massive core
        np.array([ 2.0, -0.5]),  # Right skew
        np.array([-0.5,  2.0])   # Top skew
    ]
    covs = [
        np.array([[ 2.0,  1.0],  # Soft, correlated core
                  [ 1.0,  2.0]]),
        np.array([[ 1.5,  0.0],  # Smoothly decaying tail right
                  [ 0.0,  0.8]]),
        np.array([[ 0.8,  0.0],  # Smoothly decaying tail top
                  [ 0.0,  1.5]])
    ]
    gmm = GaussianMixture2D(weights, means, covs)"""
  
    # -----------------------------------------------------------------
    # The "Sharp Comet" (High-Frequency + Dense Matrix)
    # -----------------------------------------------------------------
    '''weights = [0.15, 0.50, 0.35]
    means = [
        np.array([ 2.5,  2.5]),  # The "Spike": Sharp, high-frequency off-center peak
        np.array([ 0.0,  0.0]),  # The "Core": Asymmetric, strongly correlated base
        np.array([-2.0, -1.5])   # The "Tail": Broad, skewed spread
    ]
    covs = [
        np.array([[ 0.05, 0.02],   # Extremely tight variance -> forces high frequencies
                  [ 0.02, 0.05]]),
        np.array([[ 1.20, 0.80],   # Correlated, fills off-diagonals (no checkerboard)
                  [ 0.80, 1.50]]),
        np.array([[ 2.50, -0.50],  # Broad background, provides heavy asymmetry
                  [-0.50,  2.00]])
    ]
    gmm = GaussianMixture2D(weights, means, covs)
    '''
    # -----------------------------------------------------------------
    # High-frequency Hermite probe distribution
    # -----------------------------------------------------------------
    distrib = OscillatoryGaussian2D(
        omega=3.7,        # puts energy around Hermite degrees ~ 10--16
        beta1=1.25,
        beta2=0.85,
        beta3=0.45,
        proposal_mult=80,
    )

    p_data_eval = distrib.sample(n_eval, rng)

    if pinf_only:
        plot_hermite_pdata_and_theory_limits_row(
            p_data_eval=p_data_eval,
            alphas=[0.1, 0.5, 0.9],
            t0=t0,
            max_degree=max_hermite_degree,
            outdir=outdir,
            remove_constant=True,
        )
        return

    else:
    # p_data_eval = gmm.sample(n_eval, rng)

        alpha_dict = defaultdict(list)

        for alpha in alphas:
            
            '''p_inf = sample_neumann_limit(gmm, alpha, t0, n_eval, rng)
            exact_iterates = sample_exact_errorfree_iterates(
                gmm, alpha, t0, n_eval, n_generations, rng
            )'''

            p_inf = sample_neumann_limit_dist(distrib, alpha, t0, n_eval, rng)
            exact_iterates = sample_exact_errorfree_iterates_dist(
                distrib, alpha, t0, n_eval, n_generations, rng
            )

            # recursive learned loop
            learned_iterates = [p_data_eval.copy()]
            current_synth = None
            all_losses = []

            for gen in range(n_generations):
                print("\n" + "-" * 60)
                print(f"Generation {gen+1}/{n_generations}")

                '''if gen == 0:
                    train_data = gmm.sample(n_train, rng)
                else:
                    n_fresh = int(alpha * n_train)
                    n_synth = n_train - n_fresh
                    fresh = gmm.sample(n_fresh, rng)
                    idx = rng.choice(len(current_synth), size=n_synth, replace=True)
                    synth = current_synth[idx]
                    train_data = np.concatenate([fresh, synth], axis=0)
                    rng.shuffle(train_data)'''
                if gen == 0:
                    train_data = distrib.sample(n_train, rng)
                else:
                    n_fresh = int(alpha * n_train)
                    n_synth = n_train - n_fresh

                    fresh = distrib.sample(n_fresh, rng)

                    idx = rng.choice(len(current_synth), size=n_synth, replace=True)
                    synth = current_synth[idx]

                    train_data = np.concatenate([fresh, synth], axis=0)
                    rng.shuffle(train_data)

                train_tensor = torch.tensor(train_data, dtype=torch.float32)

                model = ScoreNet2D(hidden=128, tdim=64)
                ema, losses = train_score_network_2d(
                    model=model,
                    data_tensor=train_tensor,
                    device=device,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    lr=lr,
                    ema_decay=ema_decay,
                    t0=t0,
                    T=T,
                )
                all_losses.append(losses)

                '''samples = sample_reverse_sde_2d(
                    ema.shadow,
                    n_samples=n_eval,
                    device=device,
                    t0=t0,
                    T=T,
                    n_steps=n_steps_sample,
                    batch_size=2000,
                )
                current_synth = samples.numpy()
                learned_iterates.append(current_synth.copy())'''

                samples = sample_reverse_sde_2d(
                    ema.shadow,
                    n_samples=n_eval,
                    device=device,
                    t0=t0,
                    T=T,
                    n_steps=n_steps_sample,
                    batch_size=2000,
                )
                # --- FIX: Filter out numerical outliers (rogue particles) ---
                # Keep only samples within a reasonable bounding box (e.g., [-15, 15])
                valid_idx = torch.max(torch.abs(samples), dim=1)[0] < 15.0
                samples = samples[valid_idx]
                
                # If we lost a few samples, it won't affect the density estimations much
                current_synth = samples.numpy()
                learned_iterates.append(current_synth.copy())

                del model, ema
                if "cuda" in device:
                    torch.cuda.empty_cache()

            alpha_dict[alpha] = {}

            alpha_dict[alpha] = {}

            alpha_dict[alpha]["losses"] = all_losses
            alpha_dict[alpha]["learned_iterates"] = learned_iterates
            alpha_dict[alpha]["exact_iterates"] = exact_iterates
            alpha_dict[alpha]["p_inf"] = p_inf
            
            # -----------------------------------------------------------------
            # Distances
            # -----------------------------------------------------------------
            mu_inf, cov_inf = mean_cov(p_inf)
            mu_data, cov_data = mean_cov(p_data_eval)

            d_to_inf_w2 = []
            d_to_data_w2 = []
            d_to_inf_mmd = []
            d_exact_to_inf_w2 = []
            d_to_inf_sliced_w2 = []
            d_to_limit_exact_sliced_w2 = []

            for x_learn, x_exact in zip(learned_iterates, exact_iterates):
                mu, cov = mean_cov(x_learn)
                d_to_inf_w2.append(gaussian_w2(mu, cov, mu_inf, cov_inf))
                d_to_data_w2.append(gaussian_w2(mu, cov, mu_data, cov_data))
                d_to_inf_mmd.append(mmd_rbf(x_learn[:3000], p_inf[:3000]))
                d_to_inf_sliced_w2.append(sliced_w2(x_learn[:3000], p_inf[:3000]))

                mu_e, cov_e = mean_cov(x_exact)
                d_exact_to_inf_w2.append(gaussian_w2(mu_e, cov_e, mu_inf, cov_inf))
                d_to_limit_exact_sliced_w2.append(sliced_w2(x_exact[:3000], p_inf[:3000]))

            alpha_dict[alpha]["distance_to_limit_gaussian_w2"] = d_to_inf_w2
            alpha_dict[alpha]["distance_to_data_gaussian_w2"] = d_to_data_w2
            alpha_dict[alpha]["distance_to_limit_mmd2"] = d_to_inf_mmd
            alpha_dict[alpha]["distance_to_limit_exact_gaussian_w2"] = d_exact_to_inf_w2
            alpha_dict[alpha]["distance_to_limit_sliced_w2"] = d_to_inf_sliced_w2
            alpha_dict[alpha]["distance_to_limit_exact_sliced_w2"] = d_to_limit_exact_sliced_w2

        plot_hermite_heatmaps_clean(
            alpha_dict=alpha_dict,
            p_data_eval=p_data_eval,
            alphas=alphas,
            t0=t0,
            max_degree=max_hermite_degree,
            outdir=outdir,
            n_generations=n_generations,
            use_theoretical_limit=True,
        )

        plot_distance_to_limit(
            alpha_dict=alpha_dict,
            alphas=alphas,
            n_generations=n_generations,
            outdir=outdir,
            use_exact_w2=True,
            n_ot=1000,
        )

        plot_limiting_behaviors(
            dist=distrib,
            outdir=outdir,
            seed=seed + 123,
            n_samples=25000,
            n_ot=1000,
            alpha_fixed=0.5,
            t0_fixed=t0,
        )

        # -----------------------------------------------------------------
        # Figure 1: Hermite Heatmaps (3x3 Grid: Alphas vs. Data/Limit/Learned)
        # -----------------------------------------------------------------
        num_alphas = len(alphas)
        fig_h, axes_h = plt.subplots(num_alphas, 3, figsize=(15, 4.5 * num_alphas), squeeze=False)
        


        C_data = hermite_coeff_matrix_fixed_gaussian(
        p_data_eval, max_degree=max_hermite_degree
    )

        for i, alpha in enumerate(alphas):
            data = alpha_dict[alpha]


            '''C_inf = hermite_coeff_matrix(data["p_inf"], p_data_eval, max_degree=max_hermite_degree)
            
            C_last = hermite_coeff_matrix(data["learned_iterates"][-1], p_data_eval, max_degree=max_hermite_degree)'''

            C_inf = hermite_coeff_matrix_fixed_gaussian(
                data["p_inf"], max_degree=max_hermite_degree
            )

            C_last = hermite_coeff_matrix_fixed_gaussian(
                data["learned_iterates"][-1], max_degree=max_hermite_degree
            )

            vmax = max(np.max(np.abs(C_data)), np.max(np.abs(C_inf)), np.max(np.abs(C_last)))

            im0 = axes_h[i, 0].imshow(C_data, origin="lower", cmap="coolwarm",
                                    vmin=-vmax, vmax=vmax, interpolation="nearest")
            im1 = axes_h[i, 1].imshow(C_inf, origin="lower", cmap="coolwarm",
                                    vmin=-vmax, vmax=vmax, interpolation="nearest")
            im2 = axes_h[i, 2].imshow(C_last, origin="lower", cmap="coolwarm",
                                    vmin=-vmax, vmax=vmax, interpolation="nearest")

            if i == 0:
                axes_h[i, 0].set_title(r"$p_{\mathrm{data}}$")
                axes_h[i, 1].set_title(r"$p_\infty^\star$ (Limit)")
                axes_h[i, 2].set_title(rf"Learned $\hat p^{{{n_generations}}}$")

            axes_h[i, 0].set_ylabel(rf"$\alpha={alpha}$" + "\n" + r"$n_1$")

            # one colorbar for the whole row
            cbar = fig_h.colorbar(im2, ax=axes_h[i, :], location="right", shrink=0.85, pad=0.02)
            cbar.ax.tick_params(labelsize=12)

        for ax in axes_h[-1, :]:
            ax.set_xlabel(r"$n_2$")

        savefig(fig_h, outdir, "hermite_heatmaps_3x3")

        # -----------------------------------------------------------------
        # Figure 2 : Degree attenuation ratios + degree error to limit
        # -----------------------------------------------------------------
        fig_spec, axes_spec = plt.subplots(len(alphas), 2, figsize=(13, 4.5 * len(alphas)), squeeze=False)

        # C_data = hermite_coeff_matrix(p_data_eval, p_data_eval, max_degree=max_hermite_degree)

        '''C_data = hermite_coeff_matrix_fixed_gaussian(
            p_data_eval, max_degree=max_hermite_degree
        )'''
        R_data = np.ones_like(degree_energy(C_data))  # ratio to itself

        for i, alpha in enumerate(alphas):
            data = alpha_dict[alpha]
            col = cmap(i / max(len(alphas)-1, 1))

            '''C_inf = hermite_coeff_matrix(data["p_inf"], p_data_eval, max_degree=max_hermite_degree)
            C_last = hermite_coeff_matrix(data["learned_iterates"][-1], p_data_eval, max_degree=max_hermite_degree)'''

            C_inf = hermite_coeff_matrix_fixed_gaussian(
                data["p_inf"], max_degree=max_hermite_degree
            )

            C_last = hermite_coeff_matrix_fixed_gaussian(
                data["learned_iterates"][-1], max_degree=max_hermite_degree
            )

            ratio_inf = degree_ratio(C_inf, C_data)
            ratio_last = degree_ratio(C_last, C_data)
            err_last = degree_error(C_last, C_inf)

            ks = np.arange(len(ratio_inf))

            # Left: attenuation ratio
            ax = axes_spec[i, 0]
            ax.plot(ks, R_data, "k-", lw=2.5, label="Data / Data")
            ax.plot(ks, ratio_inf, "s--", color=COLORS["blue"], label=r"Limit / Data")
            ax.plot(ks, ratio_last, "o-", color=col, label=rf"Learned / Data ($\alpha={alpha}$)")
            ax.set_title(rf"Degree attenuation ratio ($\alpha={alpha}$)")
            ax.set_xlabel("Total degree $k$")
            ax.set_ylabel(r"$E_k / E_k^{\mathrm{data}}$")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=9)

            # Right: shellwise learned-to-limit error
            ax = axes_spec[i, 1]
            ax.plot(ks, err_last, "o-", color=col, label=rf"Learned vs limit ($\alpha={alpha}$)")
            ax.set_title(rf"Degree-wise error to limit ($\alpha={alpha}$)")
            ax.set_xlabel("Total degree $k$")
            ax.set_ylabel(r"$\sum_{n_1+n_2=k}(C_{n_1,n_2}^{\mathrm{learned}}-C_{n_1,n_2}^{\infty})^2$")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=9)

        plt.tight_layout()
        savefig(fig_spec, outdir, "degree_ratio_and_error")

        # -----------------------------------------------------------------
        # Figure 3: attenuation ratio across generations
        # -----------------------------------------------------------------
        fig_dyn, axes_dyn = plt.subplots(len(alphas), 1, figsize=(8, 4.2 * len(alphas)), squeeze=False)

        # C_data = hermite_coeff_matrix(p_data_eval, p_data_eval, max_degree=max_hermite_degree)

        for i, alpha in enumerate(alphas):
            data = alpha_dict[alpha]
            ax = axes_dyn[i, 0]

            # C_inf = hermite_coeff_matrix(data["p_inf"], p_data_eval, max_degree=max_hermite_degree)

            C_inf = hermite_coeff_matrix_fixed_gaussian(
                data["p_inf"], max_degree=max_hermite_degree
            )

            ratio_inf = degree_ratio(C_inf, C_data)
            ks = np.arange(len(ratio_inf))

            ax.plot(ks, ratio_inf, "k--", lw=2.5, label=r"Limit")

            gens_to_show = [0, 5, 7, len(data["learned_iterates"]) - 1]
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gens_to_show)))

            for c, g in zip(colors, gens_to_show):
                Cg = hermite_coeff_matrix_fixed_gaussian(
                    data["learned_iterates"][g], max_degree=max_hermite_degree
                )
                ratio_g = degree_ratio(Cg, C_data)
                ax.plot(ks, ratio_g, ":", color=c, alpha=0.95, label=rf"$N={g}$")

            ax.set_title(rf"Spectral attenuation across generations ($\alpha={alpha}$)")
            ax.set_xlabel("Total degree $k$")
            ax.set_ylabel(r"$E_k / E_k^{\mathrm{data}}$")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=9, ncol=2)

        plt.tight_layout()
        savefig(fig_dyn, outdir, "degree_ratio_across_generations")

        # -----------------------------------------------------------------
        # Figure 4: Distance to Limit Only
        # -----------------------------------------------------------------
        N = np.arange(n_generations + 1)
        fig_d, ax_d = plt.subplots(figsize=(8, 6))
        cmap = plt.cm.viridis
        
        for i, alpha in enumerate(alphas):
            data = alpha_dict[alpha]
            col = cmap(i / max(len(alphas)-1, 1))
            
            ax_d.plot(N, data["distance_to_limit_exact_sliced_w2"], "x--", color=col, alpha=0.5, label=rf"$\alpha={alpha}$ (Exact)")
            ax_d.plot(N, data["distance_to_limit_sliced_w2"], "s--", color=col, alpha=0.5, label=rf"$\alpha={alpha}$ (Learned)")

        ax_d.set_title(r"Sliced Wasserstein distance to limit ($p_\infty^\star$)")
        ax_d.set_xlabel("Generation $N$")
        ax_d.set_yscale("log")
        ax_d.grid(True, which="both", ls="-", alpha=0.2)
        ax_d.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        savefig(fig_d, outdir, "distance_w2_limit_only")

        # -----------------------------------------------------------------
        # Figure 5: Hermite Energy & Losses (2 Columns)
        # -----------------------------------------------------------------
        fig_el, axes_el = plt.subplots(num_alphas, 2, figsize=(12, 4.5 * num_alphas), squeeze=False)
        E_data = degree_energy(C_data)

        for i, alpha in enumerate(alphas):
            data = alpha_dict[alpha]
            col = cmap(i / max(len(alphas)-1, 1))

            # Energy
            axes_el[i, 0].plot(E_data, "k-", lw=3, label="Data")
            C_inf = hermite_coeff_matrix_fixed_gaussian(
                data["p_inf"], max_degree=max_hermite_degree
            )
            C_last = hermite_coeff_matrix_fixed_gaussian(
                data["learned_iterates"][-1], max_degree=max_hermite_degree
            )
            # C_inf = hermite_coeff_matrix(data["p_inf"], p_data_eval, max_degree=max_hermite_degree)
            axes_el[i, 0].plot(degree_energy(C_inf), "s:", color=COLORS["blue"], label=r"$p_\infty^\star$ (Limit)")
            
            # C_last = hermite_coeff_matrix(data["learned_iterates"][-1], p_data_eval, max_degree=max_hermite_degree)
            axes_el[i, 0].plot(degree_energy(C_last), "o--", color=col, label=rf"Learned ($\alpha={alpha}$)")
            
            axes_el[i, 0].set_yscale("log")
            axes_el[i, 0].set_title(rf"Hermite Energy ($\alpha={alpha}$)")
            axes_el[i, 0].set_ylabel("Energy")
            axes_el[i, 0].grid(True)
            axes_el[i, 0].legend(fontsize=9)
            if i == num_alphas - 1: axes_el[i, 0].set_xlabel("Total degree $k$")

            # Loss
            for g, loss_gen in enumerate(data["losses"]):
                axes_el[i, 1].plot(loss_gen, alpha=0.5, label=rf"G{g+1}" if g < 4 else None)
            axes_el[i, 1].set_title(rf"DSM Loss ($\alpha={alpha}$)")
            axes_el[i, 1].set_yscale("log")
            axes_el[i, 1].grid(True)
            if i == 0: axes_el[i, 1].legend(fontsize=9, ncol=2)
            if i == num_alphas - 1: axes_el[i, 1].set_xlabel("Epoch")

        plt.tight_layout()
        savefig(fig_el, outdir, "energy_and_loss")

        # -----------------------------------------------------------------
        # Figure: difference heatmaps (one colorbar per row)
        # -----------------------------------------------------------------
        K = max_hermite_degree
        mask = triangular_mask(K)

        fig_diff, axes_diff = plt.subplots(
            len(alphas), 2,
            figsize=(10, 4.5 * len(alphas)),
            squeeze=False,
            constrained_layout=True,
        )

        # C_data = hermite_coeff_matrix(p_data_eval, p_data_eval, max_degree=K)

        all_vals = []
        heatmaps = {}
        for alpha in alphas:
            data = alpha_dict[alpha]

            # C_inf = hermite_coeff_matrix(data["p_inf"], p_data_eval, max_degree=K)
            # C_last = hermite_coeff_matrix(data["learned_iterates"][-1], p_data_eval, max_degree=K)

            C_inf = hermite_coeff_matrix_fixed_gaussian(
                data["p_inf"], max_degree=max_hermite_degree
            )

            C_last = hermite_coeff_matrix_fixed_gaussian(
                data["learned_iterates"][-1], max_degree=max_hermite_degree
            )

            D1 = C_inf - C_data
            D2 = C_last - C_inf
            heatmaps[alpha] = (D1, D2)
            all_vals.extend([np.abs(D1[~mask]).max(), np.abs(D2[~mask]).max()])

        vmax = max(all_vals)

        for i, alpha in enumerate(alphas):
            D1, D2 = heatmaps[alpha]

            D1m = np.ma.array(D1, mask=mask)
            D2m = np.ma.array(D2, mask=mask)

            im1 = axes_diff[i, 0].imshow(
                D1m, origin="lower", cmap="coolwarm",
                vmin=-vmax, vmax=vmax, interpolation="nearest"
            )
            im2 = axes_diff[i, 1].imshow(
                D2m, origin="lower", cmap="coolwarm",
                vmin=-vmax, vmax=vmax, interpolation="nearest"
            )

            axes_diff[i, 0].set_title(rf"$C_\infty - C_{{\mathrm{{data}}}}$ ($\alpha={alpha}$)")
            axes_diff[i, 1].set_title(rf"$C_{{\mathrm{{learned}}}} - C_\infty$ ($\alpha={alpha}$)")

            axes_diff[i, 0].set_ylabel(r"$n_1$")

            # one colorbar for the whole row
            cbar = fig_diff.colorbar(
                im2, ax=axes_diff[i, :],
                location="right",
                fraction=0.035,
                pad=0.02
            )
            cbar.ax.tick_params(labelsize=12)

        for ax in axes_diff[-1, :]:
            ax.set_xlabel(r"$n_2$")

        savefig(fig_diff, outdir, "difference_heatmaps")

        # -----------------------------------------------------------------
        # Figure : Contour Comparison (3 Rows: Alphas, 3 Cols: Data, Final, Limit)
        # -----------------------------------------------------------------
        num_alphas = len(alphas)
        fig_c, axes_c = plt.subplots(num_alphas, 3, figsize=(15, 4.5 * num_alphas), squeeze=False)
        cmap = plt.cm.viridis

        for i, alpha in enumerate(alphas):
            data = alpha_dict[alpha]
            col = cmap(i / max(len(alphas)-1, 1))

            # Column 0: p_data
            axes_c[i, 0].scatter(p_data_eval[:1800, 0], p_data_eval[:1800, 1], s=2, alpha=0.05, color=COLORS["dark"])
            kde_contour(axes_c[i, 0], p_data_eval[:5000], color=COLORS["dark"], levels=8)
            if i == 0: axes_c[i, 0].set_title(r"$p_{\mathrm{data}}$")
            axes_c[i, 0].set_ylabel(rf"$\alpha={alpha}$" + "\n" + r"$x_2$")

            # Column 1: Final Learned Iterates (p^N)
            samp_last = data["learned_iterates"][-1]
            axes_c[i, 1].scatter(samp_last[:1800, 0], samp_last[:1800, 1], s=2, alpha=0.05, color=col)
            kde_contour(axes_c[i, 1], samp_last[:5000], color=col, levels=8)
            if i == 0: axes_c[i, 1].set_title(rf"Learned $\hat p^{{{n_generations}}}$")

            # Column 2: The Neumann Limit p_inf
            p_inf = data["p_inf"]
            axes_c[i, 2].scatter(p_inf[:1800, 0], p_inf[:1800, 1], s=2, alpha=0.05, color=COLORS["blue"])
            kde_contour(axes_c[i, 2], p_inf[:5000], color=COLORS["blue"], levels=8)
            if i == 0: axes_c[i, 2].set_title(r"$p_\infty^\star$ (Limit)")

        for ax in axes_c.flat:
            ax.set_xlim(-7.0, 7.0); ax.set_ylim(-4.0, 4.0)
            ax.set_aspect("equal"); ax.grid(True)
        
        for ax in axes_c[-1, :]:
            ax.set_xlabel(r"$x_1$")

        plt.tight_layout()
        savefig(fig_c, outdir, "contour_summary_3x3")

        # -----------------------------------------------------------------
        # Figure: Fourier power heatmaps (one colorbar per row)
        # -----------------------------------------------------------------
        num_alphas = len(alphas)
        fig_f, axes_f = plt.subplots(
            num_alphas, 3,
            figsize=(15, 4.5 * num_alphas),
            squeeze=False,
            constrained_layout=True,
        )

        xlim = (-7, 7)
        ylim = (-4, 4)
        bins = 256

        # Data once
        H_data, _, _ = density_on_grid(p_data_eval, xlim=xlim, ylim=ylim, bins=bins)
        P_data = log_power(fft_power_from_density(H_data))

        all_vals = [P_data]
        fourier_cache = {}

        for alpha in alphas:
            data = alpha_dict[alpha]
            H_inf, _, _ = density_on_grid(data["p_inf"], xlim=xlim, ylim=ylim, bins=bins)
            H_last, _, _ = density_on_grid(data["learned_iterates"][-1], xlim=xlim, ylim=ylim, bins=bins)

            P_inf = log_power(fft_power_from_density(H_inf))
            P_last = log_power(fft_power_from_density(H_last))

            fourier_cache[alpha] = (P_inf, P_last)
            all_vals.extend([P_inf, P_last])

        vmin = min(np.min(A) for A in all_vals)
        vmax = max(np.max(A) for A in all_vals)

        for i, alpha in enumerate(alphas):
            P_inf, P_last = fourier_cache[alpha]

            im0 = axes_f[i, 0].imshow(P_data, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            im1 = axes_f[i, 1].imshow(P_inf, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            im2 = axes_f[i, 2].imshow(P_last, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)

            if i == 0:
                axes_f[i, 0].set_title(r"$p_{\mathrm{data}}$")
                axes_f[i, 1].set_title(r"$p_\infty^\star$ (Limit)")
                axes_f[i, 2].set_title(rf"Learned $\hat p^{{{n_generations}}}$")

            axes_f[i, 0].set_ylabel(rf"$\alpha={alpha}$")

            # one colorbar for the whole row
            cbar = fig_f.colorbar(
                im2, ax=axes_f[i, :],
                location="right",
                fraction=0.035,
                pad=0.02
            )
            cbar.ax.tick_params(labelsize=12)

        for ax in axes_f[-1, :]:
            ax.set_xlabel("Fourier grid")

        savefig(fig_f, outdir, "fourier_power_heatmaps_3x3")

        # -----------------------------------------------------------------
        # Figure: Fourier difference heatmaps (one colorbar per row)
        # -----------------------------------------------------------------
        '''fig_fd, axes_fd = plt.subplots(
            len(alphas), 2,
            figsize=(10, 4.5 * len(alphas)),
            squeeze=False,
            constrained_layout=True)'''

        diff_cache = {}
        vals = []

        for alpha in alphas:
            P_inf, P_last = fourier_cache[alpha]
            D1 = P_inf - P_data
            D2 = P_last - P_inf
            diff_cache[alpha] = (D1, D2)
            vals.extend([np.max(np.abs(D1)), np.max(np.abs(D2))])

        vmax = max(vals)

        for i, alpha in enumerate(alphas):
            D1, D2 = diff_cache[alpha]

            im1 = axes_fd[i, 0].imshow(D1, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            im2 = axes_fd[i, 1].imshow(D2, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)

            axes_fd[i, 0].set_title(rf"$\log P_\infty - \log P_{{\mathrm{{data}}}}$ ($\alpha={alpha}$)")
            axes_fd[i, 1].set_title(rf"$\log P_{{\mathrm{{learned}}}} - \log P_\infty$ ($\alpha={alpha}$)")
            axes_fd[i, 0].set_ylabel(rf"$\alpha={alpha}$")

            # one colorbar for the whole row
            cbar = fig_fd.colorbar(
                im2, ax=axes_fd[i, :],
                location="right",
                fraction=0.035,
                pad=0.02
            )
            cbar.ax.tick_params(labelsize=12)

        for ax in axes_fd[-1, :]:
            ax.set_xlabel("Fourier grid")

        savefig(fig_fd, outdir, "fourier_difference_heatmaps")

        # -----------------------------------------------------------------
        # Metrics Export
        # -----------------------------------------------------------------
        serializable = {str(a): {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                        for k, v in alpha_dict[a].items() if "iterates" not in k} for a in alphas}
        with open(os.path.join(outdir, "metrics_sweep.json"), "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"\nDone. All plots saved to: {outdir}/")

if __name__ == "__main__":
    run_experiment(
        outdir="oscillatory_hermite_probe_v2",
        seed=1,
        alphas=[0.1, 0.5, 0.9],
        t0=0.10,                 # stronger than 0.05; clearer attenuation
        T=3.0,
        n_generations=20,
        n_train=40_000,
        n_eval=30_000,
        n_epochs=300,
        batch_size=512,
        lr=2e-3,
        ema_decay=0.995,
        n_steps_sample=500,
        device="cpu",
        max_hermite_degree=20,
        pinf_only=True
    )