"""
Train the generation-0 score network on CIFAR-10 only.

The resulting checkpoint is loaded by recursive_cifar10_training.py through
CFG["initial_ckpt"] (keys "score_dsm" and "ema_d").

Usage:
    python train_generation0.py --device cuda:0 --seed 45
"""
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
from scipy import linalg
from tqdm import tqdm

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
    t0             = 1e-3,
    base_ch        = 96,
    # ── training ──────────────────────────────────────────────────────────────
    epochs         = 350,
    batch_size     = 128,
    lr             = 2e-4,
    lr_min         = 1e-5,
    warmup_steps   = 3000,
    clip_grad      = 1.0,
    ema_decay      = 0.9999,
    weight_decay   = 0.01,
    min_snr_gamma  = 5.0,
    # ── evaluation ────────────────────────────────────────────────────────────
    n_gen          = 30_000,
    dpm_steps      = 50,
    eval_every     = 10,
    # ── misc ──────────────────────────────────────────────────────────────────
    num_workers    = 4,
    out_dir        = "generation0",
)

_pa = argparse.ArgumentParser()
_pa.add_argument("--device", default=os.environ.get(
    "TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
_pa.add_argument("--seed", type=int, default=45)
_pa.add_argument("--out_dir", default=CFG["out_dir"])
_args = _pa.parse_args()
device = torch.device(_args.device)
print(f"Device: {device}")

torch.manual_seed(_args.seed); torch.cuda.manual_seed_all(_args.seed); np.random.seed(_args.seed)
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

def randt(n):
    t0 = CFG["t0"]
    return torch.rand(n, device=device) * (CFG["T_sde"] - t0) + t0

def q_sample(x0, t):
    a, s = alpha(t)[:, None], sigma(t)[:, None]
    eps  = torch.randn_like(x0)
    return a * x0 + s * eps, eps

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
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
class DequantizeOnly(nn.Module):
    """Add uniform noise in [0, 1/255] to avoid quantization artifacts in FID."""
    def forward(self, x):
        # x is in [0, 1] from ToTensor
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

train_ds = torchvision.datasets.CIFAR10(
    CFG["data_dir"], train=True, download=True, transform=_train_tf
)
ref_ds = torchvision.datasets.CIFAR10(
    CFG["data_dir"], train=True, download=False, transform=_ref_tf
)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=CFG["batch_size"],
    shuffle=True,
    drop_last=True,
    pin_memory=(device.type == "cuda"),
    num_workers=CFG["num_workers"],
)

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
    sqrtm = linalg.sqrtm(s1 @ s2)
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
#  SCORE NETWORK  (identical to recursive_cifar10_training.py)
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

# ══════════════════════════════════════════════════════════════════════════════
#  DPM-SOLVER++
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def dpm_solver_pp_sample(model, n, n_steps=25, x_init=None):
    model.eval()
    ts = torch.linspace(CFG["T_sde"], CFG["t0"], n_steps + 1, device=device)

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
#  CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(path, epoch, global_step, score_dsm, ema_d, opt_d, hist, best_fid_d):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "score_dsm": score_dsm.state_dict(),
        "ema_d": ema_d.shadow,
        "opt_d": opt_d.state_dict(),
        "hist": dict(hist),
        "best_fid_d": best_fid_d,
    }
    # Atomic write: save to tmp then rename. Prevents partial files on crash.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"    Checkpoint saved → {path}")

def get_lr(global_step, total_steps):
    w = CFG["warmup_steps"]
    hi, lo = CFG["lr"], CFG["lr_min"]
    if global_step < w: return hi * global_step / max(1, w)
    p = (global_step - w) / max(1, total_steps - w)
    return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * p))

# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train(base_ch, out_dir):
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)

    score_dsm = ScoreNet(base_ch, CFG["n_res"], CFG["dropout"]).to(device)
    n_p = sum(p.numel() for p in score_dsm.parameters() if p.requires_grad)
    print(f"  base_ch={base_ch}  ScoreNet={n_p/1e6:.1f}M")

    ema_d = EMA(score_dsm, CFG["ema_decay"])
    opt_d = optim.AdamW(score_dsm.parameters(), lr=0.0, weight_decay=CFG["weight_decay"])
    scaler_d = make_scaler()

    total_steps = CFG["epochs"] * len(train_loader)
    global_step = 0
    hist = defaultdict(list)
    best_fid_d = float("inf")
    FIXED_FID_NOISE = torch.randn(
        CFG["n_gen"], D,
        generator=torch.Generator().manual_seed(1234),   # reproducible across runs
    )
    t0_wall = time.time()

    for epoch in tqdm(range(1, CFG["epochs"] + 1), desc=f"ch={base_ch}"):
        score_dsm.train()
        ep_ld = 0.0
        n_batches = 0

        for x0, _ in train_loader:
            x0 = x0.to(device, non_blocking=True)
            lr = get_lr(global_step, total_steps)
            for pg in opt_d.param_groups: pg["lr"] = lr
            global_step += 1

            t = randt(len(x0))
            xt, eps = q_sample(x0, t)

            opt_d.zero_grad(set_to_none=True)
            with amp():
                eps_pred_d = score_dsm(xt, t)
                w_snr = min_snr_weight(t, CFG["min_snr_gamma"]).detach()
                loss_d = (w_snr * ((eps_pred_d - eps) ** 2).mean(-1)).mean()
            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(opt_d)
            nn.utils.clip_grad_norm_(score_dsm.parameters(), CFG["clip_grad"])
            scaler_d.step(opt_d); scaler_d.update()
            ema_d.update(score_dsm)
            ep_ld += loss_d.item()
            n_batches += 1

        hist["epoch"].append(epoch)
        hist["loss_dsm"].append(ep_ld / n_batches)

        if epoch % 25 == 0:
            print(f"  [{base_ch}] Ep {epoch:3d}  ld={ep_ld/n_batches:.4f}  "
                  f"wall={time.time()-t0_wall:.0f}s")

        # ── Eval: FID of the EMA network, keep the best checkpoint ────────────
        if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
            with ema_d.scope(score_dsm):
                fid_d = compute_fid(
                    dpm_solver_pp_sample(score_dsm, CFG["n_gen"], CFG["dpm_steps"],
                                         x_init=FIXED_FID_NOISE)
                )
            hist["fid_epoch"].append(epoch)
            hist["fid_dsm"].append(fid_d)
            print(f"  [{base_ch}] Eval Ep {epoch}: FID_DSM={fid_d:.2f}")

            save_checkpoint(out / "ckpt_last.pt", epoch, global_step,
                            score_dsm, ema_d, opt_d, hist, best_fid_d)
            if fid_d < best_fid_d:
                best_fid_d = fid_d
                save_checkpoint(out / "ckpt_best.pt", epoch, global_step,
                                score_dsm, ema_d, opt_d, hist, best_fid_d)

            with open(out / "hist.json", "w") as f:
                json.dump(hist, f, indent=2)

    print(f"Done. Best FID_DSM={best_fid_d:.2f}  → {out / 'ckpt_best.pt'}")


if __name__ == "__main__":
    train(CFG["base_ch"], Path(_args.out_dir) / f"seed{_args.seed}")
