import math, time, json, os, argparse
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from scipy import linalg

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    data_dir       = "./data",
    n_res          = 2,
    time_dim       = 256,
    dropout        = 0.1,
    beta_min       = 0.1,
    beta_max       = 20.0,
    T_sde          = 1.0,
    t0             = 1e-3, # was 1e-2
    # ── training ──────────────────────────────────────────────────────────────
    epochs         = 300,
    batch_size     = 128,
    lr             = 2e-4,
    lr_min         = 1e-5,
    warmup_steps   = 3000,
    clip_grad      = 1.0,
    ema_decay      = 0.9999,
    weight_decay   = 0.01,
    # ── Model Collapse ────────────────────────────────────────────────────────
    n_generations  = 30,
    alpha_real     = 0.1,
    synthetic_size = 50_000,
    save_every_gen = True,
    initial_ckpt   = "generation0/seed45/ckpt_best.pt",   # written by train_generation0.py
    # ── evaluation ────────────────────────────────────────────────────────────
    n_gen          = 30_000,
    dpm_steps      = 50,    # was 25, but increased for better stability in FID estimates
    eval_every     = 50,
    n_eval_batches = 50,
    min_snr_gamma = 5.0,
    # ── sweep ─────────────────────────────────────────────────────────────────
    sweep_channels = [96],
    seeds = [45, 47],
    # ── misc ──────────────────────────────────────────────────────────────────
    num_workers    = 4,
    out_dir        =  "MULTISEED_v24_sn96_beta20_ramp20_75_b256_dpm50_t0e3_alpha01",
    ckpt_every_epoch = 20,            # [NEW] save checkpoint every N epochs
    seed           = 42,
)

CFG.update(dict(
    t0_train      = 1e-5,         # train at the smallest t_0 you might sample at
    t0_init       = 1e-2,         # t_0^{(0)} for annealed schedule
    schedule      = "fixed",   # "fixed" or "annealed"
    schedule_beta = 1.0,          # decay exponent
))

def t0_at_generation(i):
    """Returns t_0^{(i)} used for SAMPLING from generation i's network."""
    if CFG["schedule"] == "fixed":
        return CFG["t0"]
    return max(CFG["t0_init"] / (1 + i) ** CFG["schedule_beta"], CFG["t0_train"])

_pa = argparse.ArgumentParser(add_help=False)
_pa.add_argument("--device", default=os.environ.get(
    "TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
_args, _ = _pa.parse_known_args()
device = torch.device(_args.device)
print(f"Device: {device}")


torch.manual_seed(CFG["seed"]); torch.cuda.manual_seed_all(CFG["seed"])
Path(CFG["out_dir"]).mkdir(exist_ok=True)
C, H, W = 3, 32, 32
D = C * H * W

_amp_ok = device.type == "cuda"
_amp_dt = (torch.bfloat16 if (_amp_ok and torch.cuda.is_bf16_supported())
           else torch.float16)

def amp():
    return (torch.amp.autocast("cuda", _amp_dt) if _amp_ok else nullcontext())

def make_scaler():
    return torch.amp.GradScaler(enabled=(_amp_ok and _amp_dt == torch.float16))

# ══════════════════════════════════════════════════════════════════════════════
#  VP-SDE
# ══════════════════════════════════════════════════════════════════════════════
def _int_beta(t):
    return CFG["beta_min"] * t + 0.5 * (CFG["beta_max"] - CFG["beta_min"]) * t ** 2

def alpha(t):   return torch.exp(-0.5 * _int_beta(t))
def sigma(t):   return (1.0 - torch.exp(-_int_beta(t)) + 1e-8).sqrt()

def q_sample(x0, t):
    a, s = alpha(t)[:, None], sigma(t)[:, None]
    eps  = torch.randn_like(x0)
    return a * x0 + s * eps, eps

# ══════════════════════════════════════════════════════════════════════════════
#  DATA + RECURSIVE MIXTURE
# ══════════════════════════════════════════════════════════════════════════════

class DequantizeOnly(nn.Module):
    """Add uniform noise in [0, 1/255]."""
    def forward(self, x):
        return (x + torch.rand_like(x) / 255.0).clamp(0, 1)


_train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    DequantizeOnly(),
    T.Lambda(lambda x: (2.0 * x - 1.0).reshape(-1)),
])

_ref_tf = T.Compose([
    T.ToTensor(),
    DequantizeOnly(),
    T.Lambda(lambda x: (2.0 * x - 1.0).reshape(-1)),
])

_eval_tf = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: (2.0 * x - 1.0).reshape(-1)),
])


train_ds = torchvision.datasets.CIFAR10(
    CFG["data_dir"], train=True, download=True, transform=_train_tf
)

ref_ds = torchvision.datasets.CIFAR10(
    CFG["data_dir"], train=True, download=False, transform=_ref_tf
)

eval_ds = torchvision.datasets.CIFAR10(
    CFG["data_dir"], train=True, download=False, transform=_eval_tf
)


class FlatTensorDataset(torch.utils.data.Dataset):
    """
    Synthetic samples already stored as flattened tensors in [-1, 1].
    """
    def __init__(self, x):
        self.x = x.float().cpu()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], 0


class RecursiveMixtureDataset(torch.utils.data.Dataset):
    """
    q_i = alpha * p_data + (1 - alpha) * p_hat_i.

    Real CIFAR-10 samples keep the original training transform:
        RandomHorizontalFlip + ToTensor + DequantizeOnly + flatten to [-1, 1].

    Synthetic samples are assumed to already be flattened in [-1, 1].
    """
    def __init__(self, real_ds, synthetic_samples, alpha=0.5, length=None):
        assert synthetic_samples is not None
        self.real_ds = real_ds
        self.synthetic_ds = FlatTensorDataset(synthetic_samples)
        self.alpha = alpha
        self.length = length if length is not None else len(real_ds)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.rand(()) < self.alpha:
            j = torch.randint(len(self.real_ds), ()).item()
            x, _ = self.real_ds[j]
            return x, 0

        j = torch.randint(len(self.synthetic_ds), ()).item()
        x, _ = self.synthetic_ds[j]
        return x, 0


_lkw = dict(
    batch_size=CFG["batch_size"],
    shuffle=True,
    drop_last=True,
    pin_memory=(device.type == "cuda"),
    num_workers=CFG["num_workers"],
)


def make_recursive_train_loader(synthetic_samples):
    mix_ds = RecursiveMixtureDataset(
        real_ds=train_ds,
        synthetic_samples=synthetic_samples,
        alpha=CFG["alpha_real"],
        length=len(train_ds),
    )

    return torch.utils.data.DataLoader(mix_ds, **_lkw)

@torch.no_grad()
def generate_synthetic_dataset(score_dsm, ema_d, n_samples, batch_size=256, t0_sample=None):
    samples = []

    with ema_d.scope(score_dsm):
        score_dsm.eval()

        for start in tqdm(
            range(0, n_samples, batch_size),
            desc="Generating synthetic data",
        ):
            n = min(batch_size, n_samples - start)

            x = dpm_solver_pp_sample(
                score_dsm,
                n=n,
                n_steps=CFG["dpm_steps"],
                x_init=None,
                t0_sample=t0_sample,
            )

            samples.append(x.cpu())

    return torch.cat(samples, dim=0)

def min_snr_weight(t, gamma=5.0):
    """
    Min-SNR-γ loss weighting from Hang et al. 2023.
    Returns per-sample weight w(t) = min(SNR(t), gamma) / SNR(t).
    For VP-SDE: SNR(t) = alpha(t)^2 / sigma(t)^2.
    """
    a = alpha(t)
    s = sigma(t).clamp(min=1e-3)
    snr = (a / s) ** 2
    return torch.clamp(snr, max=gamma) / snr   # shape (B,)

# ══════════════════════════════════════════════════════════════════════════════
#  FID
# ══════════════════════════════════════════════════════════════════════════════
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        m = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        self.net = nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3, nn.MaxPool2d(3, 2),
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3, nn.MaxPool2d(3, 2),
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d,
            m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
            m.Mixed_7a, m.Mixed_7b, m.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        # x in [-1, 1]
        x = (x + 1.0) / 2.0                         # to [0, 1]
        x = F.interpolate(x, (299, 299), mode="bilinear",
                        align_corners=False, antialias=True)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.net(x).flatten(1)

inception_net = InceptionV3().to(device)

@torch.no_grad()
def inception_features(flat_imgs, bs=128):
    imgs = flat_imgs.view(-1, C, H, W)
    return np.concatenate([
        inception_net(imgs[i:i+bs].to(device)).cpu().float().numpy()
        for i in range(0, len(imgs), bs)
    ])

def _fid_stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)

def _fid_from_stats(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    sqrtm, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(sqrtm).all():
        sqrtm = linalg.sqrtm((s1 + eps*np.eye(len(s1))) @ (s2 + eps*np.eye(len(s2))))
    if np.iscomplexobj(sqrtm): sqrtm = sqrtm.real
    return float(diff @ diff + np.trace(s1 + s2 - 2.0 * sqrtm))

print("Pre-computing InceptionV3 real statistics …")
_ref_loader = torch.utils.data.DataLoader(ref_ds, batch_size=256, num_workers=4)
_real_feats = np.concatenate([inception_features(imgs) for imgs, _ in _ref_loader])
real_mu, real_sg = _fid_stats(_real_feats); del _real_feats
print(f"  Done.  ‖μ_real‖ = {np.linalg.norm(real_mu):.3f}")

def compute_fid(flat_samples):
    feats = inception_features(flat_samples.cpu())
    m, s = _fid_stats(feats)
    return _fid_from_stats(real_mu, real_sg, m, s)


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE NETWORK
# ══════════════════════════════════════════════════════════════════════════════
class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) *
                         torch.arange(half, device=t.device) / (half - 1))
        e = t[:, None] * freq[None]
        return torch.cat([e.sin(), e.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, ic, oc, td, dropout=0.0):
        super().__init__()
        gn_ic = min(32, ic); gn_oc = min(32, oc)
        self.b1 = nn.Sequential(nn.GroupNorm(gn_ic, ic), nn.SiLU(),
                                nn.Conv2d(ic, oc, 3, padding=1))
        self.tp = nn.Sequential(nn.SiLU(), nn.Linear(td, oc))
        self.b2 = nn.Sequential(nn.GroupNorm(gn_oc, oc), nn.SiLU(),
                                nn.Dropout(dropout),
                                nn.Conv2d(oc, oc, 3, padding=1))
        self.sk = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x, te):
        h = self.b1(x) + self.tp(te)[:, :, None, None]
        return self.b2(h) + self.sk(x)

class SelfAttn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        B, Ck, Hk, Wk = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, Ck, Hk * Wk).unbind(1)
        a = (q.transpose(-2, -1) @ k * Ck ** -0.5).softmax(-1)
        return x + self.proj((v @ a.transpose(-2, -1)).reshape(B, Ck, Hk, Wk))

class ScoreNet(nn.Module):
    def __init__(self, base_ch=128, n_res=2, dropout=0.0):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 2]
        td = CFG["time_dim"] * 4
        self.temb = nn.Sequential(SinEmb(CFG["time_dim"]),
            nn.Linear(CFG["time_dim"], td), nn.SiLU(), nn.Linear(td, td))
        self.head = nn.Conv2d(C, ch[0], 3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock(ch[0], ch[0], td, dropout) for _ in range(n_res)])
        self.dwn1 = nn.Conv2d(ch[0], ch[0], 3, stride=2, padding=1)
        self.enc2 = nn.ModuleList([ResBlock(ch[0] if i==0 else ch[1], ch[1], td, dropout) for i in range(n_res)])
        self.att2 = SelfAttn(ch[1])
        self.dwn2 = nn.Conv2d(ch[1], ch[1], 3, stride=2, padding=1)
        self.enc3 = nn.ModuleList([ResBlock(ch[1] if i==0 else ch[2], ch[2], td, dropout) for i in range(n_res)])
        self.att3 = SelfAttn(ch[2])
        self.dwn3 = nn.Conv2d(ch[2], ch[2], 3, stride=2, padding=1)
        self.mid = nn.ModuleList([ResBlock(ch[2], ch[2], td, dropout), SelfAttn(ch[2]),
                                  ResBlock(ch[2], ch[2], td, dropout)])
        self.up3 = nn.ConvTranspose2d(ch[2], ch[2], 4, stride=2, padding=1)
        self.dec3 = nn.ModuleList([ResBlock(ch[2]*2 if i==0 else ch[2], ch[2], td, dropout) for i in range(n_res)])
        self.dat3 = SelfAttn(ch[2])
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 4, stride=2, padding=1)
        self.dec2 = nn.ModuleList([ResBlock(ch[1]*2 if i==0 else ch[1], ch[1], td, dropout) for i in range(n_res)])
        self.dat2 = SelfAttn(ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 4, stride=2, padding=1)
        self.dec1 = nn.ModuleList([ResBlock(ch[0]*2 if i==0 else ch[0], ch[0], td, dropout) for i in range(n_res)])
        self.out = nn.Sequential(nn.GroupNorm(min(32, ch[0]), ch[0]), nn.SiLU(),
                                 nn.Conv2d(ch[0], C, 3, padding=1))

    def forward(self, x, t):
        te = self.temb(t)
        h = self.head(x.view(-1, C, H, W))
        for b in self.enc1: h = b(h, te)
        s1 = h; h = self.dwn1(h)
        for b in self.enc2: h = b(h, te)
        h = self.att2(h); s2 = h; h = self.dwn2(h)
        for b in self.enc3: h = b(h, te)
        h = self.att3(h); s3 = h; h = self.dwn3(h)
        h = self.mid[0](h, te); h = self.mid[1](h); h = self.mid[2](h, te)
        h = self.up3(h); h = torch.cat([h, s3], 1)
        for b in self.dec3: h = b(h, te)
        h = self.dat3(h)
        h = self.up2(h); h = torch.cat([h, s2], 1)
        for b in self.dec2: h = b(h, te)
        h = self.dat2(h)
        h = self.up1(h); h = torch.cat([h, s1], 1)
        for b in self.dec1: h = b(h, te)
        return self.out(h).view(-1, D)

def randt_force(n):
    return torch.rand(n, device=device) * (CFG["T_sde"] - CFG["t0_train"]) + CFG["t0_train"]

@torch.no_grad()
def dpm_solver_pp_sample(model, n, n_steps=25, x_init=None, t0_sample=None):
    if t0_sample is None:
        t0_sample = CFG["t0_train"]
    ts = torch.linspace(CFG["T_sde"], t0_sample, n_steps + 1, device=device)

    if x_init is None:
        xt = torch.randn(n, D, device=device)
    else:
        xt = x_init.to(device).clone()

    x0_prev = None
    for i in range(n_steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        with amp():
            eps = model(xt, t_cur.expand(n))

        # No clamping inside the loop — preserves second-order solver accuracy
        x0_cur = (xt - sigma(t_cur) * eps) / alpha(t_cur)

        # Safety rail only if model has gone catastrophically off: clamp early steps
        # to prevent NaN propagation. Threshold is generous.
        if x0_cur.abs().max() > 10.0:
            x0_cur = x0_cur.clamp(-10.0, 10.0)

        if i == 0 or x0_prev is None:
            xt = alpha(t_nxt) * x0_cur + sigma(t_nxt) * (xt - alpha(t_cur) * x0_cur) / sigma(t_cur)
        else:
            r = (t_cur - t_nxt) / (ts[i - 1] - t_cur + 1e-8)
            x0_ext = (1 + 0.5 * r) * x0_cur - 0.5 * r * x0_prev
            xt = alpha(t_nxt) * x0_ext + sigma(t_nxt) * (xt - alpha(t_cur) * x0_ext) / sigma(t_cur)

        x0_prev = x0_cur

    return xt.clamp(-1, 1).cpu()

# ══════════════════════════════════════════════════════════════════════════════
#  EMA
# ══════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)
    @contextmanager
    def scope(self, model):
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        dev = next(model.parameters()).device
        model.load_state_dict({k: v.to(dev) for k, v in self.shadow.items()})
        try:    yield
        finally: model.load_state_dict(backup)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINTING  [NEW]
# ══════════════════════════════════════════════════════════════════════════════

def save_generation_checkpoint(
    path,
    generation,
    epoch,
    score_dsm,
    ema_d,
    opt_d,
    scaler_d,
    hist,
    global_step,
    cfg,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generation": generation,
        "epoch": epoch,
        "global_step": global_step,
        "score_dsm": score_dsm.state_dict(),
        "ema_d": ema_d.shadow,
        "opt_d": opt_d.state_dict(),
        "scaler_d": scaler_d.state_dict() if scaler_d else None,
        "hist": dict(hist),
        "cfg": dict(cfg),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"Checkpoint saved → {path}")


def load_generation_checkpoint(path, score_dsm, ema_d, opt_d, scaler_d):
    ck = torch.load(path, map_location=device, weights_only=False)

    score_dsm.load_state_dict(ck["score_dsm"])

    if "ema_d" in ck and ck["ema_d"] is not None:
        ema_d.shadow = {k: v.to(device) for k, v in ck["ema_d"].items()}

    if "opt_d" in ck and ck["opt_d"] is not None:
        opt_d.load_state_dict(ck["opt_d"])

    if ck.get("scaler_d") is not None and scaler_d is not None:
        scaler_d.load_state_dict(ck["scaler_d"])

    hist = defaultdict(list)
    for k, vv in ck.get("hist", {}).items():
        hist[k] = list(vv)

    if ck.get("rng_cpu") is not None:
        torch.set_rng_state(ck["rng_cpu"].cpu())

    if ck.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])

    if ck.get("rng_numpy") is not None:
        np.random.set_state(ck["rng_numpy"])

    return {
        "start_epoch": int(ck["epoch"]) + 1,
        "global_step": int(ck.get("global_step", 0)),
        "hist": hist,
    }

def load_score_and_ema_from_ckpt(path, base_ch):
    score = ScoreNet(base_ch, CFG["n_res"], CFG["dropout"]).to(device)
    ema = EMA(score, CFG["ema_decay"])

    ck = torch.load(path, map_location=device, weights_only=False)

    if "score_dsm" in ck:
        score.load_state_dict(ck["score_dsm"])
    elif "model" in ck:
        score.load_state_dict(ck["model"])
    elif "state_dict" in ck:
        score.load_state_dict(ck["state_dict"])
    else:
        raise KeyError(f"Could not find score network weights in {path}")

    if "ema_d" in ck:
        ema.shadow = {k: v.to(device) for k, v in ck["ema_d"].items()}
    elif "ema" in ck:
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
    else:
        print("Warning: no EMA found in checkpoint; using raw model weights as EMA.")
        ema = EMA(score, CFG["ema_decay"])

    return score, ema

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def get_lr(global_step, total_steps):
    w = CFG["warmup_steps"]
    hi, lo = CFG["lr"], CFG["lr_min"]
    if global_step < w: return hi * global_step / max(1, w)
    p = (global_step - w) / max(1, total_steps - w)
    return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * p))

def score_residual(eps_pred, eps_true, t):
    return (eps_true - eps_pred) / sigma(t)[:, None].clamp(min=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN ONE PAIR (DSM + SOSM) — ORIGINAL V18 LOOP + CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════════
def train_dsm_generation(base_ch, generation, train_loader, out_dir, resume_path=None):
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)

    score_dsm = ScoreNet(base_ch, CFG["n_res"], CFG["dropout"]).to(device)
    ema_d = EMA(score_dsm, CFG["ema_decay"])

    opt_d = optim.AdamW(
        score_dsm.parameters(),
        lr=0.0,
        weight_decay=CFG["weight_decay"],
    )

    scaler_d = make_scaler()

    total_steps = CFG["epochs"] * len(train_loader)
    global_step = 0
    hist = defaultdict(list)
    start_epoch = 1
    t0_wall = time.time()

    if resume_path is not None and Path(resume_path).is_file():
        print(f"[gen={generation}] Resuming from {resume_path}")
        info = load_generation_checkpoint(
            resume_path,
            score_dsm,
            ema_d,
            opt_d,
            scaler_d,
        )
        start_epoch = info["start_epoch"]
        global_step = info["global_step"]
        hist = info["hist"]

        if start_epoch > CFG["epochs"]:
            print(f"[gen={generation}] Already completed.")
            ckpt_path = Path(resume_path)
            return score_dsm, ema_d, ckpt_path, dict(hist)

    for epoch in tqdm(
        range(start_epoch, CFG["epochs"] + 1),
        initial=start_epoch - 1,
        total=CFG["epochs"],
        desc=f"gen={generation}, ch={base_ch}",
    ):
        score_dsm.train()
        ep_ld = 0.0
        n_batches = 0

        for x0, _ in train_loader:
            x0 = x0.to(device, non_blocking=True)
            n = len(x0)

            lr = get_lr(global_step, total_steps)
            for pg in opt_d.param_groups:
                pg["lr"] = lr
            global_step += 1

            t = randt_force(n)
            xt, eps = q_sample(x0, t)

            opt_d.zero_grad(set_to_none=True)

            with amp():
                eps_pred_d = score_dsm(xt, t)
                w_snr = min_snr_weight(t, CFG["min_snr_gamma"]).detach()
                loss_d = (w_snr * ((eps_pred_d - eps) ** 2).mean(-1)).mean()

            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(opt_d)

            nn.utils.clip_grad_norm_(
                score_dsm.parameters(),
                CFG["clip_grad"],
            )

            scaler_d.step(opt_d)
            scaler_d.update()
            ema_d.update(score_dsm)

            ep_ld += loss_d.item()
            n_batches += 1

        mean_loss = ep_ld / max(1, n_batches)

        hist["epoch"].append(epoch)
        hist["loss_dsm"].append(mean_loss)

        if epoch % 25 == 0 or epoch == CFG["epochs"]:
            print(
                f"[gen={generation}] Ep {epoch:3d} "
                f"loss={mean_loss:.4f} "
                f"wall={time.time() - t0_wall:.0f}s"
            )
        if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
            with ema_d.scope(score_dsm):
                samples = dpm_solver_pp_sample(
                    score_dsm,
                    n=CFG["n_gen"],
                    n_steps=CFG["dpm_steps"],
                    x_init=None,
                    t0_sample=t0_at_generation(generation),
                )
                fid = compute_fid(samples)

            hist["fid_epoch"].append(epoch)
            hist["fid_dsm"].append(fid)

            print(
                f"[gen={generation}] Eval Ep {epoch:3d} "
                f"FID={fid:.2f}"
            )

        latest_path = out / f"generation_{generation:03d}_latest.pt"
        save_generation_checkpoint(
            latest_path,
            generation=generation,
            epoch=epoch,
            score_dsm=score_dsm,
            ema_d=ema_d,
            opt_d=opt_d,
            scaler_d=scaler_d,
            hist=hist,
            global_step=global_step,
            cfg=CFG,
        )

    ckpt_path = out / f"generation_{generation:03d}_epoch_{CFG['epochs']}.pt"

    save_generation_checkpoint(
        ckpt_path,
        generation=generation,
        epoch=CFG["epochs"],
        score_dsm=score_dsm,
        ema_d=ema_d,
        opt_d=opt_d,
        scaler_d=scaler_d,
        hist=hist,
        global_step=global_step,
        cfg=CFG,
    )

    with open(out / f"generation_{generation:03d}_hist.json", "w") as f:
        json.dump(
            {
                k: [
                    float(x) if isinstance(x, (float, np.floating))
                    else int(x) if isinstance(x, (int, np.integer))
                    else x
                    for x in vv
                ]
                for k, vv in hist.items()
            },
            f,
            indent=2,
        )

    return score_dsm, ema_d, ckpt_path, dict(hist)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    out = Path(CFG["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    base_ch = CFG["sweep_channels"][0]
    seed = CFG["seeds"][0]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    generation_ckpts = []

    gen0_synth_path = out / f"synthetic_samples_from_generation_000.pt"

    if gen0_synth_path.exists():
        print(f"Loading cached generation-0 synthetic samples from {gen0_synth_path}")
        synthetic_samples = torch.load(gen0_synth_path, map_location="cpu", weights_only=False)
    else:
        print(f"Loading generation-0 checkpoint from {CFG['initial_ckpt']}")
        score_prev, ema_prev = load_score_and_ema_from_ckpt(
            CFG["initial_ckpt"],
            base_ch=base_ch,
        )

        t0_gen0 = t0_at_generation(0)
        print(f"Generating synthetic samples from pretrained generation 0 at t0={t0_gen0:.2e}...")
        synthetic_samples = generate_synthetic_dataset(
            score_prev,
            ema_prev,
            n_samples=CFG["synthetic_size"],
            batch_size=CFG["batch_size"],
            t0_sample=t0_gen0,
        )

        torch.save(synthetic_samples, gen0_synth_path)

        del score_prev, ema_prev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


    for generation in range(1, CFG["n_generations"] + 1):
        print("\n" + "#" * 80)
        print(f"Training recursive generation {generation}")
        print("#" * 80)

        train_loader_i = make_recursive_train_loader(synthetic_samples)

        gen_out = out / f"ch{base_ch}" / f"seed{seed}" / f"generation_{generation:03d}"

        latest_path = gen_out / f"generation_{generation:03d}_latest.pt"
        final_path = gen_out / f"generation_{generation:03d}_epoch_{CFG['epochs']}.pt"

        resume_path = None
        if final_path.exists():
            resume_path = final_path
        elif latest_path.exists():
            resume_path = latest_path

        score_i, ema_i, ckpt_path, hist = train_dsm_generation(
            base_ch=base_ch,
            generation=generation,
            train_loader=train_loader_i,
            out_dir=gen_out,
            resume_path=resume_path,
        )

        generation_ckpts.append(str(ckpt_path))

        synth_path = gen_out / f"synthetic_samples_from_generation_{generation:03d}.pt"

        if synth_path.exists():
            print(f"Loading cached synthetic samples from {synth_path}")
            synthetic_samples = torch.load(synth_path, map_location="cpu", weights_only=False)
        else:
            t0_i = t0_at_generation(generation)
            print(f"Generating synthetic samples from generation {generation} at t0={t0_i:.2e}...")
            synthetic_samples = generate_synthetic_dataset(
                score_i,
                ema_i,
                n_samples=CFG["synthetic_size"],
                batch_size=CFG["batch_size"],
                t0_sample=t0_i,
            )
            torch.save(synthetic_samples, synth_path)

        del score_i, ema_i, train_loader_i

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    with open(out / "recursive_checkpoints.json", "w") as f:
        json.dump(generation_ckpts, f, indent=2)

if __name__ == "__main__":
    main()