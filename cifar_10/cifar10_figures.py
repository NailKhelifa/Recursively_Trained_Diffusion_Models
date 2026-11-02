"""
CIFAR-10 figures of the paper.

    Figure 3a  fig_fid_to_spinf_gen10_shifted.pdf   FID(p^N, p^11) over generations
    Figure 3b  fig_hf_deficit_heatmap_first10.pdf   high-frequency deficit over (alpha, N)
    Figure 5   fig_truncationvf.pdf                 effect of t0 on the error-free limit p_inf^*

Figures 3a and 3b read the synthetic samples written by recursive_cifar10_training.py
(<run>/synthetic_samples_from_generation_000.pt and
 <run>/ch96/seed45/generation_XXX/synthetic_samples_from_generation_XXX.pt).
Figure 5 only needs CIFAR-10.

Code generation g is displayed as N = g + 1, so N = 1 is the model trained on real data only.

Usage:
    python cifar10_figures.py --runs_dir . --device cuda:0             # all figures
    python cifar10_figures.py --figures 5                               # Figure 5 only
"""
import math, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm.auto import tqdm
from torchvision.models import inception_v3, Inception_V3_Weights

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
ALPHAS   = [0.1, 0.3, 0.5, 0.7, 0.9]
RUN_NAME = "MULTISEED_v24_sn96_beta20_ramp20_75_b256_dpm50_t0e3_alpha{:02d}"
BASE_CH  = 96
SEED     = 45
T0       = 1e-3
BETA_MIN = 0.1
BETA_MAX = 20.0
N_FID    = 10000
N_SPEC   = 10000          # radial spectra (Figure 5)
N_SPEC_HF = 2000          # high-frequency energy (Figure 3b)
HF_CUTOFF = 15
C, H, W  = 3, 32, 32

ALPHA_COLORS = {
    0.1: "#d62728",
    0.3: "#ff7f0e",
    0.5: "#9467bd",
    0.7: "#1f77b4",
    0.9: "#2ca02c",
}

BASE_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 200,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "lines.markersize": 1,
    "legend.frameon": False,
    "grid.alpha": 0.5,
    "grid.linewidth": 0.5,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

# ══════════════════════════════════════════════════════════════════════════════
#  CIFAR REFERENCE SAMPLES
# ══════════════════════════════════════════════════════════════════════════════
class DequantizeOnly(torch.nn.Module):
    def forward(self, x):
        return (x + torch.rand_like(x) / 255.0).clamp(0, 1)

_ref_ds = None

def init_data(data_dir):
    global _ref_ds
    ref_tf = T.Compose([
        T.ToTensor(),
        DequantizeOnly(),
        T.Lambda(lambda x: (2.0 * x - 1.0).reshape(-1)),
    ])
    _ref_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=ref_tf)

def sample_real(n):
    idx = torch.randint(len(_ref_ds), (n,))
    return torch.stack([_ref_ds[int(i)][0] for i in idx]).float()

# ══════════════════════════════════════════════════════════════════════════════
#  ERROR-FREE LIMITING LAW
# ══════════════════════════════════════════════════════════════════════════════
def int_beta(t):
    return BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t**2

def vp_alpha(t):
    return math.exp(-0.5 * int_beta(t))

def sample_pinf(alpha_real, n, t0=T0):
    """
    Sample p_inf^* = alpha sum_{k>=0} (1-alpha)^k U_{(k+1)t0}(p_data).

    Draw M ~ Geom(alpha) on {1,2,...}
    Draw X ~ p_data, Z ~ N(0,I)
    Return a(t0)^M X + sqrt(1-a(t0)^{2M}) Z.
    """
    x = sample_real(n)
    z = torch.randn_like(x)
    m = torch.from_numpy(np.random.geometric(alpha_real, size=n)).float()
    a0 = vp_alpha(t0)
    a = torch.pow(torch.tensor(a0), m).view(-1, 1)
    return (a * x + torch.sqrt(1 - a**2) * z).clamp(-1, 1).cpu()

# ══════════════════════════════════════════════════════════════════════════════
#  SAVED SYNTHETIC SAMPLES
# ══════════════════════════════════════════════════════════════════════════════
def samples_path(root, g):
    if g == 0:
        return root / "synthetic_samples_from_generation_000.pt"
    return (root / f"ch{BASE_CH}" / f"seed{SEED}" / f"generation_{g:03d}"
            / f"synthetic_samples_from_generation_{g:03d}.pt")

def load_samples(root, g, n):
    """First n samples of generation g (files hold 50,000 samples; only n are kept)."""
    p = samples_path(root, g)
    if not p.exists():
        raise FileNotFoundError(f"Missing samples for generation {g}: {p}")
    x = torch.load(p, map_location="cpu", weights_only=False)
    return x[:n].clone().float()

# ══════════════════════════════════════════════════════════════════════════════
#  FID
# ══════════════════════════════════════════════════════════════════════════════
class InceptionFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        self.net = torch.nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3, torch.nn.MaxPool2d(3, 2),
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3, torch.nn.MaxPool2d(3, 2),
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d,
            m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
            m.Mixed_7a, m.Mixed_7b, m.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = (x + 1.0) / 2.0
        x = F.interpolate(x, (299, 299), mode="bilinear", align_corners=False, antialias=True)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        return self.net((x - mean) / std).flatten(1)

class FID:
    def __init__(self, device, cache):
        self.device = device
        self.inception = InceptionFeatures().to(device)
        if cache.exists():
            obj = torch.load(cache, weights_only=False)
            self.real_mu, self.real_sg = obj["mu"], obj["sg"]
        else:
            self.real_mu, self.real_sg = self.stats(sample_real(N_FID))
            torch.save({"mu": self.real_mu, "sg": self.real_sg}, cache)

    @torch.no_grad()
    def stats(self, flat, bs=128):
        flat = flat[:N_FID].cpu()
        out = []
        for i in tqdm(range(0, len(flat), bs), leave=False):
            x = flat[i:i+bs].view(-1, C, H, W).to(self.device)
            out.append(self.inception(x).cpu().numpy())
        feat = np.concatenate(out)
        return feat.mean(0), np.cov(feat, rowvar=False)

    @staticmethod
    def from_stats(mu1, sg1, mu2, sg2, eps=1e-6):
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sg1 @ sg2)
        if not np.isfinite(covmean).all():
            I = np.eye(sg1.shape[0])
            covmean = linalg.sqrtm((sg1 + eps * I) @ (sg2 + eps * I))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sg1 + sg2 - 2 * covmean))

    def to_real(self, x):
        mu, sg = self.stats(x)
        return self.from_stats(self.real_mu, self.real_sg, mu, sg)

# ══════════════════════════════════════════════════════════════════════════════
#  SPECTRA
# ══════════════════════════════════════════════════════════════════════════════
def radial_power(flat, nbins=24):
    x = flat[:N_SPEC].view(-1, C, H, W).float()
    gray = x.mean(1).numpy()

    fft = np.fft.fftshift(np.fft.fft2(gray, axes=(-2, -1)), axes=(-2, -1))
    power = np.abs(fft)**2
    power = power.mean(0)

    yy, xx = np.meshgrid(np.arange(H) - H//2, np.arange(W) - W//2, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2)
    bins = np.linspace(0, rr.max(), nbins + 1)

    r, s = [], []
    for i in range(nbins):
        mask = (rr >= bins[i]) & (rr < bins[i+1])
        r.append(0.5 * (bins[i] + bins[i+1]))
        s.append(power[mask].mean())
    return np.array(r), np.array(s)

def hf_energy(samples_flat, cutoff=HF_CUTOFF, n_spec=N_SPEC_HF):
    x = samples_flat[:n_spec].view(-1, C, H, W).cpu().numpy()
    gray = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]

    Nn, Hh, Ww = gray.shape
    pwr = np.zeros((Hh, Ww), dtype=np.float64)

    for img in gray:
        f = np.fft.fftshift(np.fft.fft2(img))
        pwr += np.abs(f) ** 2

    pwr /= Nn

    cy, cx = Hh // 2, Ww // 2
    yy, xx = np.ogrid[:Hh, :Ww]
    r = np.hypot(yy - cy, xx - cx)

    return float(pwr[r >= cutoff].mean())

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3a — FID(p^N, p^11)
# ══════════════════════════════════════════════════════════════════════════════
def figure_3a(roots, fid, out, ref_gen=10, n_gens=11):
    cache = out / "fid_to_spinf_gen10.pt"
    if cache.exists():
        fid_to_spinf = torch.load(cache, weights_only=False)
    else:
        fid_to_spinf = {}
        for a in ALPHAS:
            print(f"\nAlpha={a}: reference = generation {ref_gen}")
            mu_ref, sg_ref = fid.stats(load_samples(roots[a], ref_gen, N_FID))
            fid_to_spinf[a] = {}
            for n in range(n_gens):
                mu_n, sg_n = fid.stats(load_samples(roots[a], n, N_FID))
                fid_to_spinf[a][n] = fid.from_stats(mu_n, sg_n, mu_ref, sg_ref)
                print(f"  FID(p^{n}, p^{ref_gen}) = {fid_to_spinf[a][n]:.2f}")
        torch.save(fid_to_spinf, cache)

    rc = {
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
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "legend.frameon": True,
    }
    with plt.rc_context({**BASE_RC, **rc}):
        fig, ax = plt.subplots(figsize=(10, 5.8))

        for a in ALPHAS:
            # True generations 0,...,8 displayed at N = 1,...,9
            gens_true = np.arange(0, 9)
            vals = [fid_to_spinf[a][n] for n in gens_true]
            ax.plot(gens_true + 1, vals, "-o", color=ALPHA_COLORS[a], lw=2.0, ms=5.5,
                    mec="white", mew=0.7, label=fr"$\alpha={a}$", zorder=3)

        ax.set_xlabel(r"Generation $N$")
        ax.set_ylabel(r"FID$(\hat{p}^{\,N},\, \hat{p}^{\,11})$")
        ax.set_xticks(range(1, 10))
        ax.set_xlim(0.5, 8.5)
        ax.grid(True, which="major", color="0.7", alpha=0.25, lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        leg = ax.legend(loc="upper right", ncols=2, columnspacing=1.2,
                        handlelength=2.2, borderpad=0.6)
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_edgecolor("0.75")

        fig.tight_layout()
        fig.savefig(out / "fig_fid_to_spinf_gen10_shifted.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out / 'fig_fid_to_spinf_gen10_shifted.pdf'}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3b — high-frequency deficit D_HF(alpha, N) = 1 - HF(p^N) / HF(p_data)
# ══════════════════════════════════════════════════════════════════════════════
def figure_3b(roots, out, n_gens=8, vmin=0.0, vmax=0.5):
    cache = out / "hf_deficit_heatmap_first10.pt"
    if cache.exists():
        hf_deficit_mat = torch.load(cache, weights_only=False)["hf_deficit_mat"]
    else:
        hf_ref = hf_energy(sample_real(N_SPEC_HF))
        print(f"HF_real = {hf_ref:.6e}")

        hf_ratio_mat = np.full((len(ALPHAS), n_gens), np.nan, dtype=np.float64)
        for i, a in enumerate(ALPHAS):
            for g in range(n_gens):
                hf_ratio_mat[i, g] = hf_energy(load_samples(roots[a], g, N_SPEC_HF)) / hf_ref
                print(f"alpha={a}, gen={g:2d}: HF ratio = {hf_ratio_mat[i, g]:.4f}")

        hf_deficit_mat = 1.0 - hf_ratio_mat
        torch.save(
            {
                "hf_ratio_mat": hf_ratio_mat,
                "hf_deficit_mat": hf_deficit_mat,
                "alphas": ALPHAS,
                "gens_true": list(range(n_gens)),
                "gens_display": list(range(1, n_gens + 1)),
            },
            cache,
        )

    rc = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "axes.linewidth": 0.8,
    }
    with plt.rc_context({**BASE_RC, **rc}):
        fig, ax = plt.subplots(figsize=(8.8, 5.0))

        im = ax.imshow(hf_deficit_mat, origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="auto")

        gens_display = list(range(1, hf_deficit_mat.shape[1] + 1))
        ax.set_xticks(np.arange(len(gens_display)))
        ax.set_xticklabels(gens_display)
        ax.set_yticks(np.arange(len(ALPHAS)))
        ax.set_yticklabels([f"{a:.1f}" for a in ALPHAS])
        ax.set_xlabel(r"Generation $N$")
        ax.set_ylabel(r"Fresh-data proportion $\alpha$")

        for side in ["left", "bottom", "top", "right"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(axis="both", which="major", direction="in", length=2, width=0.8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label(r"High-frequency deficit", rotation=90, labelpad=12)

        fig.tight_layout()
        fig.savefig(out / "fig_hf_deficit_heatmap_first10.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out / 'fig_hf_deficit_heatmap_first10.pdf'}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — truncation time t0 controls the error-free limit p_inf^*
# ══════════════════════════════════════════════════════════════════════════════
def figure_5(fid, out):
    T0_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    alphas_to_show = [0.1, 0.5, 0.9]
    ALPHA_FOR_TRUNC = 0.5

    rc = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
    with plt.rc_context({**BASE_RC, **rc}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))

        # Panel A: FID(p_inf^*(t0), p_data) for several alpha
        ax = axes[0]
        acolors = plt.get_cmap("cividis")(np.linspace(0.15, 0.80, len(alphas_to_show)))
        for color, a in zip(acolors, alphas_to_show):
            fids = []
            for t0 in T0_GRID:
                fids.append(fid.to_real(sample_pinf(a, N_FID, t0=t0)))
                print(f"alpha={a}, t0={t0:g}: FID(p_inf, data) = {fids[-1]:.2f}")
            ax.plot(T0_GRID, fids, "-o", color=color, lw=2.0, ms=6,
                    mec="white", mew=0.6, label=fr"$\alpha={a}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Truncation time $t_0$")
        ax.set_ylabel(r"FID$(p_\infty^\star,\, p_{\mathrm{data}})$")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="upper left")

        # Panel B: radial power ratio of p_inf^* to p_data at varying t0 (one alpha)
        ax = axes[1]
        _, s_real = radial_power(sample_real(N_SPEC))
        tcolors = plt.get_cmap("magma")(np.linspace(0.15, 0.85, len(T0_GRID)))
        for color, t0 in zip(tcolors, T0_GRID):
            r, s = radial_power(sample_pinf(ALPHA_FOR_TRUNC, N_SPEC, t0=t0))
            ax.plot(r, s / (s_real + 1e-12), color=color, lw=1.7, label=fr"$t_0={t0:g}$")
        ax.axhline(1.0, color="0.3", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(r"Radial frequency $|k|$")
        ax.set_ylabel("Power ratio")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="upper left", ncols=2, columnspacing=0.9, handlelength=1.4)

        fig.tight_layout()
        fig.savefig(out / "fig_truncationvf.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out / 'fig_truncationvf.pdf'}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--runs_dir", default=".", help="folder holding the MULTISEED_* runs")
    pa.add_argument("--data_dir", default="./data")
    pa.add_argument("--out_dir", default="neurips_figures")
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--figures", nargs="+", default=["3a", "3b", "5"], choices=["3a", "3b", "5"])
    pa.add_argument("--seed", type=int, default=0)
    args = pa.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots = {a: Path(args.runs_dir) / RUN_NAME.format(int(round(10 * a))) for a in ALPHAS}

    init_data(args.data_dir)
    fid = None
    if "3a" in args.figures or "5" in args.figures:
        fid = FID(args.device, out / "real_fid_stats.pt")

    if "3a" in args.figures:
        figure_3a(roots, fid, out)
    if "3b" in args.figures:
        figure_3b(roots, out)
    if "5" in args.figures:
        figure_5(fid, out)


if __name__ == "__main__":
    main()
