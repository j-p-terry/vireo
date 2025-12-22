from contextlib import contextmanager
import kornia as K
import math
import matplotlib.pyplot as plt
import numpy as np
from timm.layers import DropBlock2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ReduceLROnPlateau, SequentialLR
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure, structural_similarity_index_measure
from typing import Tuple, Dict, Optional
import pytorch_lightning as pl
from pytorch_msssim import ms_ssim  
import random
import wandb
import time

torch.use_deterministic_algorithms(True, warn_only=True)

@contextmanager
def timer(name, stats_dict):
    t0 = time.perf_counter()
    yield
    stats_dict[name] = stats_dict.get(name, 0.0) + (time.perf_counter() - t0)

# Custom weight initialization: Kaiming normal for layers with GELU.            
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
class SelfEnsemble:

    def __init__(self,):
        # ---- D4 group: 8 aug modes ----
        # Rotations by 0/90/180/270; optionally mirror to get full 8.
        self.D4_4 = ("r0","r90","r180","r270")
        self.D4_8 = ("r0","r90","r180","r270","mr0","mr90","mr180","mr270")  # mirror+rotate

    def _apply_aug(self, img: torch.Tensor, mode: str) -> torch.Tensor:
        # img: (B, C, H, W)
        if mode == "r0":
            return img
        if mode == "r90":
            return img.rot90(1, dims=(2,3))
        if mode == "r180":
            return img.rot90(2, dims=(2,3))
        if mode == "r270":
            return img.rot90(3, dims=(2,3))
        if mode == "mr0":
            return torch.flip(img, dims=(3,))  # horizontal mirror
        if mode == "mr90":
            return torch.flip(img.rot90(1, dims=(2,3)), dims=(3,))
        if mode == "mr180":
            return torch.flip(img.rot90(2, dims=(2,3)), dims=(3,))
        if mode == "mr270":
            return torch.flip(img.rot90(3, dims=(2,3)), dims=(3,))
        raise ValueError(mode)

    def _invert_aug(self, img: torch.Tensor, mode: str) -> torch.Tensor:
        # inverse of _apply_aug
        if mode == "r0":
            return img
        if mode == "r90":
            return img.rot90(3, dims=(2,3))
        if mode == "r180":
            return img.rot90(2, dims=(2,3))
        if mode == "r270":
            return img.rot90(1, dims=(2,3))
        if mode == "mr0":
            return torch.flip(img, dims=(3,))
        if mode == "mr90":
            return img.rot90(3, dims=(2,3)).flip(dims=(3,))
        if mode == "mr180":
            return img.rot90(2, dims=(2,3)).flip(dims=(3,))
        if mode == "mr270":
            return img.rot90(1, dims=(2,3)).flip(dims=(3,))
        raise ValueError(mode)

    def _make_otf_from_psf(self, psf: torch.Tensor, use_ortho: bool = True) -> torch.Tensor:
        """
        psf: (B,1,H,W) real, centered impulse at (H//2,W//2), sum=1
        returns: (B,1,H,W//2+1) complex (rfft2 of ifftshifted PSF)
        """
        norm = "ortho" if use_ortho else "backward"
        psf0 = torch.fft.ifftshift(psf, dim=(-2,-1))
        return torch.fft.rfft2(psf0, norm=norm)

    @torch.no_grad()
    def tta_self_ensemble(
        self,
        model: pl.LightningModule,
        dirty: torch.Tensor,      # (B,1,H,W)
        psf: torch.Tensor,        # (B,1,h,w) or (B,1,H,W) centered; will be padded upstream
        use_8: bool = False,
        data_range_head: str = "out_dc",  # which output to ensemble: "out_dc" (recommended if using Weiner DC) or "out"
    ):
        """
        Runs D4 TTA: rotate/flip dirty+PSF, recompute OTF from the aug PSF, forward pass, invert, average.
        Returns averaged dict with keys like {"out": ..., "out_dc": ...}.
        """
        # model.eval()
        modes = self.D4_8 if use_8 else self.D4_4
        preds = []

        for m in modes:
            d_aug  = self._apply_aug(dirty, m)
            psf_aug = self._apply_aug(psf,   m)

            out = model(d_aug, psf_aug,)  # expects dict {"out", "out_dc"} or similar
            # Pick the field to ensemble (usually x_dc)
            y = out[data_range_head]
            y_inv = self._invert_aug(y, m)
            preds.append(y_inv)

        # Average (or median) the inversions
        y_stack = torch.stack(preds, dim=0)  # (M, B, 1, H, W)
        y_mean  = y_stack.mean(dim=0)

        out_dict = {data_range_head: y_mean}
        return out_dict
            
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param)

    @torch.no_grad()
    def apply_to(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])

class StarletLoss(nn.Module):
    """
    à trous / starlet transform with B3-spline kernel (separable 5×5).
    Computes L1/Huber distance between starlet detail coefficients across scales.
    Uses depthwise conv with increasing dilation; no zero-inserted kernels.
    """
    def __init__(
        self,
        reduction: str = "mean",
        loss: str = "l1",                # "l1" or "huber"
        huber_delta: float = 0.02,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss = loss
        self.huber_delta = huber_delta
        # 1D B3-spline kernel (normalized); we’ll form k = h ⊗ h on the fly
        h = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32) / 16.0
        self.register_buffer("_h1d", h.view(1, 1, 1, 5))  # store once; moved with module
        # simple cache: (C, dtype, device) -> 2D kernel for depthwise conv
        self._k2d_cache: Dict[Tuple[int, torch.dtype, torch.device], torch.Tensor] = {}

    def _kernel2d(self, C: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (C, dtype, device)
        k = self._k2d_cache.get(key)
        if k is not None:
            return k
        # make 2D: h⊗h, shape (1,1,5,5), then expand to depthwise (C,1,5,5)
        h1 = self._h1d.to(device=device, dtype=dtype)           # (1,1,1,5)
        h2 = self._h1d.transpose(-1, -2).to(device=device, dtype=dtype)  # (1,1,5,1)
        k2d = torch.matmul(h2, h1)                              # (1,1,5,5)
        k2d = k2d.expand(C, 1, 5, 5).contiguous()               # depthwise
        self._k2d_cache[key] = k2d
        return k2d

    @staticmethod
    def _pad_for_dilation(x: torch.Tensor, k: int, dilation: int) -> torch.Tensor:
        # same-like padding for kernel size k with dilation d
        pad = (k - 1) // 2 * dilation
        return F.pad(x, (pad, pad, pad, pad), mode="reflect")

    def _smooth(self, x: torch.Tensor, dil: int, k2d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_pad = self._pad_for_dilation(x, k=5, dilation=dil)
        return F.conv2d(x_pad, k2d, bias=None, stride=1, padding=0, dilation=dil, groups=C)

    def starlet_coeffs(self, x: torch.Tensor, n_scales: int) -> Tuple[list, torch.Tensor]:
        """
        Returns (coeffs, residual) where coeffs[j] is detail at scale j (1..n_scales).
        """
        k2d = self._kernel2d(x.shape[1], x.dtype, x.device)
        c_prev = x
        coeffs = []
        for j in range(n_scales):
            dil = 2 ** j
            c = self._smooth(c_prev, dil, k2d)
            w = c_prev - c
            coeffs.append(w)
            c_prev = c
        return coeffs, c_prev  # residual

    def _crit(self, diff: torch.Tensor) -> torch.Tensor:
        if self.loss == "l1":
            val = diff.abs()
        else:  # Huber
            d = diff.abs()
            delta = self.huber_delta
            val = torch.where(d <= delta, 0.5 * d * d / max(delta, 1e-12), d - 0.5 * delta)
        if self.reduction == "mean":
            return val.mean()
        elif self.reduction == "sum":
            return val.sum()
        return val

    def forward(self, pred: torch.Tensor, target: torch.Tensor, n_scales: int = 3,
                per_scale_weights: Optional[list] = None) -> torch.Tensor:
        """
        pred, target: (B,1 or C,H,W)
        n_scales: number of starlet scales (1..4 is typical)
        per_scale_weights: optional list length n_scales (e.g., emphasize mid/high freq)
        """
        # optional per-scale weights
        if per_scale_weights is None:
            # slightly upweight finer scales
            per_scale_weights = [1.0 + 0.15 * j for j in range(n_scales)]

        wp, _ = self.starlet_coeffs(pred, n_scales)
        wt, _ = self.starlet_coeffs(target, n_scales)

        loss = 0.0
        for j in range(n_scales):
            loss = loss + per_scale_weights[j] * self._crit(wp[j] - wt[j])
        return loss

class AdaptiveStarletRegularizer(nn.Module):
    """
    Wrapper that chooses 'full' vs 'light' starlet each step:
      - Full: n_scales_full (e.g., 3–4), gradients enabled.
      - Light: n_scales_light (e.g., 1), optional downsample, optional no_grad.
    """
    def __init__(
        self,
        n_scales_full: int = 3,
        n_scales_light: int = 1,
        full_every_k: int = 8,
        downsample_light: int = 2,   # 2→ 1/2 res; 1 → no downsample
        use_no_grad_light: bool = True,
        loss: str = "l1",
        huber_delta: float = 0.02,
    ):
        super().__init__()
        self.full = StarletLoss(loss=loss, huber_delta=huber_delta)
        self.light = StarletLoss(loss=loss, huber_delta=huber_delta)
        self.n_scales_full = n_scales_full
        self.n_scales_light = n_scales_light
        self.full_every_k = max(1, int(full_every_k))
        self.downsample_light = max(1, int(downsample_light))
        self.use_no_grad_light = use_no_grad_light

    def _maybe_down(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample_light == 1:
            return x
        H, W = x.shape[-2:]
        return F.interpolate(x, size=(H // self.downsample_light, W // self.downsample_light),
                             mode="bilinear", align_corners=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, step: int, mode: str = "train"):
        """
        Returns (loss, meta_dict)
        """
        run_full = (mode != "train") or (step % self.full_every_k == 0)

        if run_full:
            val = self.full(pred, target, n_scales=self.n_scales_full)
            meta = {
                "starlet_mode": "full",
                "n_scales": self.n_scales_full,
                "downsample": 1,
            }
        else:
            p = self._maybe_down(pred.detach() if self.use_no_grad_light else pred)
            t = self._maybe_down(target.detach() if self.use_no_grad_light else target)
            if self.use_no_grad_light:
                with torch.no_grad():
                    val = self.light(p, t, n_scales=self.n_scales_light)
                # keep the computational graph shape stable by multiplying with a scalar 1.0
                val = val * 1.0
            else:
                val = self.light(p, t, n_scales=self.n_scales_light)
            meta = {
                "starlet_mode": "light",
                "n_scales": self.n_scales_light,
                "downsample": self.downsample_light,
            }
        return val, meta

class DenoiseLoss(nn.Module):
    def __init__(self, w_main=1.0, w_ssim=0.5, w_aux=0.1, w_fwd=0.1,
                 calc_msssim_every: int = 10, w_spectral: float = 1e-3,
                 w_beam_img: float = 0.1, w_oob: float = 1e-3, 
                 w_starlet: float = 1e-3,
                 use_ortho=True,
                 schedule_loss: bool = True, 
                 ramp_epochs: int = 10,
                 tau0: float = 0.08, tau_min: float = 0.04,
                 calc_starlet_every: int = 10,
                 starlet_window: int | None = None,
                 w_low: float = 0.1,
                 w_high: float = 1e-3,
                 w_blur: float = 0.1,
                 w_psd: float = 1e-3,
                 ):
        super().__init__()
        self.w_main = w_main
        self.w_ssim = w_ssim
        self.w_aux = w_aux
        self.w_fwd = w_fwd
        self.w_spectral = w_spectral
        self.w_beam_img = w_beam_img
        self.w_oob = w_oob
        self.w_starlet = w_starlet
        self.calc_msssim_every = calc_msssim_every
        self.calc_starlet_every = calc_starlet_every
        self.starlet_window = starlet_window
        self.use_ortho = use_ortho
        self.schedule_loss = schedule_loss
        self.ramp_epochs = ramp_epochs
        self.tau0 = tau0
        self.tau_min = tau_min
        self.tau = tau0
        self.w_low = w_low
        self.w_high = w_high
        self.w_blur = w_blur
        self.w_psd = w_psd
        self.band_weights = {
            "low": w_low,
            "high": w_high,
        }
        self.band_ranges = {
            "low": 0.25,
            "high": 0.6,
        }
        self.band_weight_sum = w_low + w_high
        
        self.starlet_reg = AdaptiveStarletRegularizer(
            n_scales_full=3,         # try 3; 4 is pricier
            n_scales_light=1,        # cheap pass
            full_every_k=calc_starlet_every,          # full starlet every 8 steps
            downsample_light=1,      # compute light starlet at 1/2 res
            use_no_grad_light=True,  # avoids graph/memory bloat
            loss="huber",            # huber tends to be stabler
            huber_delta=0.02,
        )

        
        self.target_weights = {
            "w_main":     w_main,
            "w_ssim":     w_ssim,
            "w_aux":      w_aux,
            "w_beam_img": w_beam_img,
            "w_blur":     w_blur,
            "w_fwd":      w_fwd,
            "w_spectral": w_spectral,
            "w_oob":      w_oob,
            "w_starlet":  w_starlet,
            "w_psd":      w_psd,
        }
        
        self.current_weights = self.target_weights.copy()
        
        # explicit mapping from weight name -> metric key
        self.weight_to_metric = {
            "w_ssim":     "ssim_main",
            "w_aux":      "mae_aux",
            "w_beam_img": "mae_beam_img",
            "w_fwd":      "mae_fwd_fourier",
            "w_spectral": "spectral",
            "w_oob":      "oob_power",
            "w_main":     "mae_main",
            "w_starlet":  "starlet",
            "w_blur":     "blur_mae",
            "w_psd":      "psd_highk",
            
        }
        
        self.weight_sum = sum(self.current_weights.values())
        
        self.metric_to_weight = {v: k for k, v in self.weight_to_metric.items()}

    # ---------- FFT helpers ----------
    def _fft2c(self, x):  # centered orthonormal FFT
        return torch.fft.rfft2(x, norm="ortho" if self.use_ortho else "backward")

    def _ifft2c(self, X, size_hw):
        return torch.fft.irfft2(X, s=size_hw, norm="ortho" if self.use_ortho else "backward")

    def _make_otf_from_psf(self, psf, hw, eps=1e-12):
        # psf_sum1: centered PSF with unit sum (crop & recenter first)
        psf_sum1 = psf / torch.sum(psf, dim=(-2, -1), keepdim=True)
        P = torch.fft.ifftshift(psf_sum1, dim=(-2,-1))        # remove center offset
        H = torch.fft.rfft2(P, norm="backward")                # complex OTF

        # normalize DC no matter the FFT norm:
        dc = H[..., :1, :1]                       # keep dims, complex
        mag2 = (dc.real*dc.real + dc.imag*dc.imag).clamp_min(eps)  # real, safe
        inv_dc = dc.conj() / mag2                 # stable complex reciprocal

        H_unitdc = H * inv_dc                     # now DC ≈ 1+0j

        return H_unitdc

    def _ensure_otf(self, psf=None, otf_c=None, hw=None):
        """
        If psf is given: returns otf_c from psf.
        If otf_c is given: returns otf_c as complex tensor.
        """
        if otf_c is not None:
            assert torch.is_complex(otf_c), "otf_c must be a complex tensor"
            return otf_c
        elif psf is not None:
            assert hw is not None, "Must pass hw=(H,W) when creating OTF from PSF"
            return self._make_otf_from_psf(psf, hw)
        else:
            raise ValueError("Must provide either psf or otf_c")
            
    def _weight_schedule(self, epoch: int, loss_stats: dict | None = None):
        """
        Dynamically adjust loss weights each epoch.
        """
        # ----- ramp with nonzero start -----
        ramp_epochs = max(1, int(getattr(self, "ramp_epochs", 1)))
        factor = min(1.0, epoch / ramp_epochs)
        min_frac = 0.05  # 5% at epoch 0

        def ramped(target: float) -> float:
            return target * (min_frac + (1.0 - min_frac) * factor)

        targets = dict(self.target_weights)  # copy

        # set ramped weights (keep w_main fixed)
        for name, tgt in targets.items():
            if name == "w_main":
                setattr(self, name, float(tgt))
            else:
                setattr(self, name, float(ramped(tgt)))

        # ----- optional adaptive scaling -----
        if loss_stats is None:
            return {k: getattr(self, k) for k in targets}

        ref = float(loss_stats.get("mae_main", 1.0))
        ref = max(ref, 1e-8)

        adapt_strength = 0.5     # 0=no adapt, 1=full

        for w_name, metric_key in self.weight_to_metric.items():
            if w_name == "w_main":
                continue
            val = loss_stats.get(metric_key, None)
            if val is None or val <= 0:
                continue

            # scale toward mae_main’s magnitude
            scale = (ref / float(val)) ** adapt_strength

            tgt = float(targets[w_name])
            cur = float(getattr(self, w_name))
            new_w = cur * scale

            # clamp relative to target
            lo, hi = 0.1 * tgt, 5.0 * tgt
            if not math.isfinite(new_w):
                new_w = tgt  # fallback
            new_w = min(hi, max(lo, new_w))

            setattr(self, w_name, new_w)
            
        self.current_weights = {k: getattr(self, k) for k in targets}

        return self.current_weights
                    
    # ---------- Forward ----------
    def forward(self, recon, x_dc, clean, dirty, psf=None, otf_c=None, batch_idx=None, phase: str = "train"):
        # --- Main MAE loss ---
        mae_main = F.l1_loss(x_dc, clean)
        
        pred_small   = F.interpolate(x_dc, scale_factor=0.5, mode='bilinear', align_corners=False)
        target_small = F.interpolate(clean, scale_factor=0.5, mode='bilinear', align_corners=False)

        # --- MS-SSIM ---
        if batch_idx is None or batch_idx % self.calc_msssim_every == 0 or phase != "train":
            ssim_main = 1.0 - ms_ssim(clean, x_dc, data_range=2.0, size_average=True)
        else:
            ssim_main = 1.0 - ms_ssim(target_small, pred_small, data_range=2.0, size_average=True)

        # --- Combine ---
        loss_total = (
            self.w_main * mae_main +
            self.w_ssim * ssim_main
        )

        loss_dict = {
            "mae_main": mae_main.item(),
            "ssim_main": ssim_main.item(),
            "base_loss": mae_main.item() + ssim_main.item(),
        }
        
        B, _, H, W = x_dc.shape
        
        # --- Ensure OTF (complex) ---
        if getattr(self, "w_fwd", 0.0) > 0.0 or getattr(self, "w_oob", 0.0) > 0.0:
            otf_c = self._ensure_otf(psf=psf, otf_c=otf_c, hw=(H, W))
            if otf_c.shape[0] == 1 and B > 1:
                otf_c = otf_c.expand(B, -1, -1, -1)
            Hk = otf_c.squeeze(1)
        
        if getattr(self, "w_aux", 0.0) > 0.0:
            mae_aux = F.l1_loss(recon, clean)
            loss_total = loss_total + self.w_aux * mae_aux
            loss_dict["mae_aux"] = mae_aux.item()
        
        if getattr(self, "w_fwd", 0.0) > 0.0:
            # --- Fourier forward loss ---
            Xk = self._fft2c(x_dc.squeeze(1))
            Yk = self._fft2c(dirty.squeeze(1))
            mae_fwd = torch.abs(Hk * Xk - Yk).mean()
            loss_total = loss_total + self.w_fwd * mae_fwd
            loss_dict["mae_fwd_fourier"] = mae_fwd.item()

        if getattr(self, "w_oob", 0.0) > 0.0:
            # --- Out-of-band suppression ---
            Hmag = torch.abs(Hk)
            k = 30.0
            if batch_idx == 0:
                self.tau = max(self.tau_min, min(self.tau0, self.tau * 0.99))
            soft_mask = torch.sigmoid(k * (self.tau - Hmag))
            outband_power = (soft_mask * (torch.abs(Xk)**2)).mean()
            loss_total = loss_total + self.w_oob * outband_power
            loss_dict["oob_power"] = outband_power.item()
        
        if getattr(self, "w_blur", 0.0) > 0.0:
            # --- Blur loss ---
            S_true_blur = self.blur_with_psf(clean, psf)        # noise-free restored target
            S_hat_blur  = self.blur_with_psf(x_dc,  psf)

            L_blur = F.l1_loss(S_hat_blur, S_true_blur)  
            loss_total = loss_total + self.w_blur * L_blur
            loss_dict["blur_mae"] = L_blur.item()
        
        if getattr(self, "w_beam_img", 0.0) > 0.0:
            # --- Beam forward loss (image space) ---
            # psf: (B,1,H,W) centered, peak=1
            psf0   = torch.fft.ifftshift(psf.squeeze(1), dim=(-2,-1))   # (B,H,W)
            Hk_img = torch.fft.rfft2(psf0, norm="ortho").unsqueeze(1)   # (B,1,H,Wr)

            # Xk is FFT of prediction in the same norm/size
            Xk = torch.fft.rfft2(x_dc, norm="ortho")                   # (B,1,H,Wr)
            x_conv = torch.fft.irfft2(Hk_img * Xk, s=(H,W), norm="ortho").real
            mae_beam_img = F.l1_loss(x_conv, dirty)
            loss_total = loss_total + self.w_beam_img * mae_beam_img
            loss_dict["mae_beam_img"] = mae_beam_img.item()

        if getattr(self, "w_spectral", 0.0) > 0.0:
            bands = []
            for band in self.band_weights:
                if self.band_weights[band] > 0.0:
                    bands.append([band, self.band_ranges[band]])
            spec = self.spectral_mag_loss_restored(x_dc, clean, bands=bands)
            loss_total = loss_total + self.w_spectral * spec
            loss_dict["spectral"] = max(1e-5, spec.item())
            
        if getattr(self, "w_psd", 0.0) > 0.0:
            psd = self.psd_highk_loss(x_dc, clean)
            loss_total = loss_total + self.w_psd * psd
            loss_dict["psd_highk"] = psd.item()
            
        if getattr(self, "w_starlet", 0.0) > 0.0:
            # --- Starlet loss ---
            starlet_loss, _ = self.starlet_reg(pred_small, target_small, step=batch_idx, mode=phase)
            loss_total = loss_total + self.w_starlet * starlet_loss
            loss_dict["starlet"] = starlet_loss.item()

        return loss_total, loss_dict
    
    # 2) small forward-consistency in restored domain
    def blur_with_psf(self, x, psf_centered_peak1):
        H, W = x.shape[-2:]
        Hk = torch.fft.rfft2(torch.fft.ifftshift(psf_centered_peak1, dim=(-2,-1)), norm="ortho")
        Xk = torch.fft.rfft2(x.squeeze(1), norm="ortho")
        y  = torch.fft.irfft2(Hk * Xk, s=(H, W), norm="ortho")
        return y.unsqueeze(1)
    
    def psd_highk_loss(self, pred, target, k0=0.4, eps=1e-8):
        # pred/target: (B,1,H,W)
        win_y = torch.hann_window(pred.shape[-2], device=pred.device).view(1,1,-1,1)
        win_x = torch.hann_window(pred.shape[-1], device=pred.device).view(1,1,1,-1)
        win   = win_y * win_x

        Pk = torch.fft.rfft2((pred - pred.mean(dim=(-2,-1), keepdim=True))*win, norm="ortho").abs()
        Tk = torch.fft.rfft2((target - target.mean(dim=(-2,-1), keepdim=True))*win, norm="ortho").abs()

        B, C, H, Wr = Pk.shape
        W = 2*(Wr-1)
        fy = torch.fft.fftfreq(H, d=1., device=pred.device).abs()[:,None]
        fx = torch.fft.rfftfreq(W, d=1., device=pred.device).abs()[None,:]
        r  = torch.sqrt(fy*fy + fx*fx)  # normalized radius (0..1)

        # per-sample spectral shape normalization
        Pn = Pk / (Pk.amax(dim=(-2,-1), keepdim=True) + eps)
        Tn = Tk / (Tk.amax(dim=(-2,-1), keepdim=True) + eps)

        mask = (r >= k0).to(Pn.dtype)
        L = (torch.log1p(Pn) - torch.log1p(Tn)).abs() * mask
        return L.mean()

    def spectral_mag_loss_restored(self, pred, target, bands=[("low", 0.25), ("high", 0.6)], 
                                   eps_rel=1e-6,):
        """
        pred, target: (B,1,H,W) restored-domain images
        band: ("low", fc) | ("high", fc) | ("band", (f_lo, f_hi)), freqs in [0..1] of Nyquist
        """
        Xk = torch.fft.rfft2(pred.squeeze(1),   norm="ortho")
        Yk = torch.fft.rfft2(target.squeeze(1), norm="ortho")

        B, H, Wr = Xk.shape
        W = 2*(Wr - 1)

        # radial frequencies normalized to Nyquist
        ky = torch.fft.fftfreq(H, d=1.0).to(pred.device).abs()[:, None]
        kx = torch.fft.rfftfreq(W, d=1.0).to(pred.device).abs()[None, :]
        r  = torch.sqrt(ky*ky + kx*kx) / 0.5  # 0..1
        
        for i, band in enumerate(bands):

            if band[0] == "low":
                M = (r <= band[1]).float()
            elif band[0] == "high":
                M = (r >= band[1]).float()
            elif band[0] == "band":
                lo, hi = band[1]
                M = ((r >= lo) & (r <= hi)).float()
            else:
                raise ValueError("band must be ('low', fc), ('high', fc), or ('band', (flo,fhi))")

            magX = Xk.abs()
            magY = Yk.abs()

            # epsilon tied to target amplitude so scale is anchored
            eps = eps_rel * magY.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

            # log-magnitude difference (phase-robust)
            if i == 0:
                L = self.band_weights[band[0]] * (M * (torch.log1p(magX/eps) - torch.log1p(magY/eps)).abs()).mean()
            else:
                L = L + self.band_weights[band[0]] * (M * (torch.log1p(magX/eps) - torch.log1p(magY/eps)).abs()).mean()
        return L / self.band_weight_sum

class SoftClipHead(nn.Module):
    def __init__(self, init_a: float = 2.0, min_a: float = 0.1):
        super().__init__()
        # we parametrize a = softplus(raw_a) + min_a to keep it > min_a
        self.raw_a = nn.Parameter(torch.log(torch.exp(torch.tensor(init_a - min_a)) - 1.0))
        self.min_a = min_a

    @property
    def a(self):
        # ensure positivity and a >= min_a
        return F.softplus(self.raw_a) + self.min_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a
        return a * torch.tanh(x / a)
    
class AsinhHead(nn.Module):
    def __init__(self, init_k: float = 1.0, min_k: float = 1e-3):
        super().__init__()
        # raw parameters, we’ll reparametrize to keep them positive:
        self.raw_k = nn.Parameter(torch.log(torch.exp(torch.tensor(init_k - min_k)) - 1.))
        self.min_k = min_k

    @property
    def k(self):
        return F.softplus(self.raw_k) + self.min_k

    def forward(self, x):
        k = self.k
        return k * torch.asinh(x / k)
    
    
class SlidingWindowMHSA(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, patch_size: int = 8, shift: bool = False):
        """
        channels: feature channels
        num_heads: attention heads
        patch_size: size of square patch (window)
        shift: whether to shift the patch grid by patch_size // 2
        """
        super().__init__()
        assert channels % num_heads == 0
        self.C = channels
        self.H = num_heads
        self.Ch = channels // num_heads
        self.patch_size = patch_size
        self.shift = shift

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        ph = self.patch_size
        assert H % ph == 0 and W % ph == 0, \
            f"Input ({H},{W}) must be divisible by patch_size={ph}"

        # Optionally shift the input for overlapping attention windows
        if self.shift:
            shift_amt = ph // 2
            x = torch.roll(x, shifts=(-shift_amt, -shift_amt), dims=(-2, -1))

        # chunk into non-overlapping patches
        nH, nW = H // ph, W // ph
        x_patches = x.view(B, C, nH, ph, nW, ph) \
                      .permute(0, 2, 4, 1, 3, 5) \
                      .reshape(B * nH * nW, C, ph, ph)

        # QKV projection
        qkv = self.qkv(x_patches)  # (B*nH*nW, 3C, ph, ph)
        Np = ph * ph
        qkv = qkv.view(B * nH * nW, 3, self.H, self.Ch, Np)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each: (B*nH*nW, heads, Ch, Np)

        # scaled dot-product attention
        scores = torch.matmul(q.permute(0, 1, 3, 2), k) / math.sqrt(self.Ch)
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.matmul(v, attn.permute(0, 1, 3, 2))

        # merge heads
        out = out.view(B * nH * nW, C, ph, ph)
        out = self.proj(out)

        # reconstruct image from patches
        out = out.view(B, nH, nW, C, ph, ph) \
                 .permute(0, 3, 1, 4, 2, 5) \
                 .reshape(B, C, H, W)

        # shift back if we shifted input
        if self.shift:
            shift_amt = ph // 2
            out = torch.roll(out, shifts=(shift_amt, shift_amt), dims=(-2, -1))

        return self.gamma * out + x

class SlidingWindowAttentionBlock(nn.Module):
    """
    Combines normal and shifted attention for boundary-aware coverage
    """
    def __init__(self, channels, num_heads=4, patch_size=8):
        super().__init__()
        self.attn1 = SlidingWindowMHSA(channels, num_heads, patch_size, shift=False)
        self.attn2 = SlidingWindowMHSA(channels, num_heads, patch_size, shift=True)

    def forward(self, x):
        # average normal + shifted attention outputs
        out1 = self.attn1(x)
        out2 = self.attn2(x)
        return 0.5 * (out1 + out2)
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must divide num_heads"
        self.C = channels
        self.H = num_heads
        self.Ch = channels // num_heads

        # one conv to produce Q, K, V
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        # final projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # produce combined QKV and split
        qkv = self.qkv(x)                    # B, 3C, H, W
        qkv = qkv.view(B, 3, self.H, self.Ch, N)  # B, 3, heads, Ch, N
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]    # each: B, H, Ch, N

        # scaled dot-product per head
        # (B, H, N, Ch) × (B, H, Ch, N) → (B, H, N, N)
        scores = torch.matmul(q.permute(0,1,3,2), k) / math.sqrt(self.Ch)
        attn   = torch.softmax(scores, dim=-1)

        # (B, H, Ch, N) ← (B, H, Ch, N) × (B, H, N, N)
        out = torch.matmul(v, attn.permute(0,1,3,2))  # B, H, Ch, N

        # merge heads back
        out = out.view(B, C, H, W)  # reshape outputs back to spatial

        # final 1×1 projection and residual
        out = self.proj(out)
        return self.gamma * out + x

class SelfAttention(nn.Module):
    """
    Light self-attention block for spatial feature maps.
    """
    def __init__(self, channels: int, window_size: int = 0):
        super().__init__()
        self.window_size = window_size ## 19 is a good option for 76x76
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def windowed_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        # 1) pad to multiple of ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))  # (B,C,Hp,Wp)
        Hp, Wp = H + pad_h, W + pad_w

        # 2) cyclic shift
        shift = ws // 2
        x_shift = torch.roll(x_pad, shifts=(-shift, -shift), dims=(2,3))

        # 3) project to Q,K,V on the shifted feature map
        q_s = self.q(x_shift)  # (B,C,Hp,Wp)
        k_s = self.k(x_shift)
        v_s = self.v(x_shift)

        # 4) slice into windows by reshaping: (B, C, Hp//ws, ws, Wp//ws, ws)
        #    then bring window dims forward
        def make_windows(tensor):
            B,C,Hp,Wp = tensor.shape
            return (
                tensor
                .view(B, C, Hp // ws, ws, Wp // ws, ws)
                .permute(0,2,4,1,3,5)    # (B, nh, nw, C, ws, ws)
                .reshape(-1, C, ws * ws) # (B*nh*nw, C, ws*ws)
            )

        qw = make_windows(q_s)
        kw = make_windows(k_s)
        vw = make_windows(v_s)

        # 5) per-window attention
        #    qw: (M,C,L), kw: (M,C,L) → attn: (M, L, L)
        #    then out_windows: (M, C, L)
        L = ws * ws
        attn = torch.softmax(torch.bmm(qw.transpose(1,2), kw), dim=-1)
        out_w = torch.bmm(vw, attn.transpose(1,2))  # (M, C, L)

        # 6) fold back into spatial grid
        #    reshape to (B, nh, nw, C, ws, ws)
        nh, nw = Hp // ws, Wp // ws
        out = (
            out_w
            .view(B, nh, nw, C, ws, ws)
            .permute(0,3,1,4,2,5)      # (B, C, nh, ws, nw, ws)
            .reshape(B, C, Hp, Wp)     # (B, C, Hp, Wp)
        )

        # 7) reverse shift & crop
        out = torch.roll(out, shifts=(shift, shift), dims=(2,3))
        return out[:, :, :H, :W]

    # @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.window_size == 0:
            q = self.q(x).reshape(B, C, H*W).transpose(1,2)  # B,N,C
            k = self.k(x).reshape(B, C, H*W)                 # B,C,N
            v = self.v(x).reshape(B, C, H*W)                 # B,C,N
            attn = torch.softmax(torch.bmm(q, k), dim=-1)    # B,N,N
            out = torch.bmm(attn, v.transpose(1,2)).transpose(1,2).reshape(B, C, H, W)
        else:
            out = self.windowed_attention(x)
        return self.gamma * out + x
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        An Inception block that applies parallel 1x1, 3x3, and 5x5 convolutions,
        concatenates the results, and fuses them with a 1x1 convolution.
        """
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, padding_mode="reflect")
        # Fuse the concatenated branches back to out_channels.
        self.conv_out = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()
    
    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = torch.cat([b1, b3, b5], dim=1)
        out = self.conv_out(out)
        out = self.act(out)
        return out

class ResidualBlock(nn.Module):
    """
    A basic residual block with optional DropBlock: two conv layers with GroupNorm and GELU,
    plus a residual skip (with projection if needed).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.0,
        dropblock_prob: float = 0.0,    
        dropblock_size: int = 7,
        num_groups: int = 8,
        dilation: int = 1,
    ):
        super().__init__()
        padding = dilation * (kernel_size-1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode="reflect", dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.act1 = nn.GELU()
        self.drop1 = DropBlock2d(block_size=dropblock_size, drop_prob=dropblock_prob) if dropblock_prob>0 else nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode="reflect", dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.act2 = nn.GELU()
        self.drop2 = DropBlock2d(block_size=dropblock_size, drop_prob=dropblock_prob) if dropblock_prob>0 else nn.Dropout2d(dropout)

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels!=out_channels else nn.Identity()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        if self.training:
            out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        if self.training:
            out = self.drop2(out)
        return out + identity
  
class InceptionResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 bottleneck_channels=None, dropout=0.0,
                 dropblock_prob=0.0, dropblock_size=7,
                 add_skip: bool = True,
                 ):
        super().__init__()
        # optional 1×1 reduction
        b = bottleneck_channels or in_channels // 2
        self.reduce = nn.Conv2d(in_channels, b, kernel_size=1)
        
        # parallel paths
        self.conv1 = nn.Conv2d(b, out_channels//3, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(b, out_channels//3, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv5 = nn.Conv2d(b, out_channels - 2*(out_channels//3), 
                               kernel_size=5, padding=2, padding_mode="reflect")
        
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), 
                                 num_channels=out_channels)
        self.act  = nn.GELU()
        if dropblock_prob > 0 or dropout > 0:
            self.drop = DropBlock2d(block_size=dropblock_size, drop_prob=dropblock_prob) if dropblock_prob>0 else nn.Dropout2d(dropout)
        else:
            self.drop = nn.Identity()
        
        # projection if needed to match dims
        self.proj = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())
        
        self.add_skip = add_skip

    def forward(self, x):
        identity = self.proj(x)
        y = self.reduce(x)
        p1 = self.conv1(y)
        p3 = self.conv3(y)
        p5 = self.conv5(y)
        out = torch.cat([p1, p3, p5], dim=1)
        out = self.norm(out)
        out = self.act(out)
        if self.training:
            out = self.drop(out)
        return out if not self.add_skip else out + identity

# --- geometry from centered, peak-1 PSF (dirty or restoring) ---
@torch.no_grad()
def psf_geom_features(psf_centered: torch.Tensor, frac: float = 0.5):
    """
    psf_centered: (B,1,H,W) or (B,H,W), peak at center (peak=1).
    Returns (B, 6): [log1p(FWHM_maj_px), log1p(FWHM_min_px), axial_ratio, cos(2θ), sin(2θ), log1p(px_per_beam)]
    θ is BPA east-of-north (encoded as 2θ to resolve 180° ambiguity).
    """
    x = psf_centered
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    B, _, H, W = x.shape
    dev, dt = x.device, x.dtype
    eps = torch.finfo(dt).eps

    # Peak-normalize for geometry (scale invariance)
    p = x

    # Coordinate grids (dtype match)
    cy, cx = H // 2, W // 2
    yy = torch.arange(H, device=dev, dtype=dt) - cy
    xx = torch.arange(W, device=dev, dtype=dt) - cx
    YY, XX = torch.meshgrid(yy, xx, indexing="ij")

    # Core mask at given fraction of peak (half-max by default)
    core = (p >= frac).to(dt)
    # Ignore negative sidelobes entirely for geometry
    w = torch.clamp(p, min=0) * core

    s = w.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    mx = (w * XX).sum(dim=(-2, -1), keepdim=True) / s
    my = (w * YY).sum(dim=(-2, -1), keepdim=True) / s

    DX, DY = XX - mx, YY - my
    vxx = (w * DX * DX).sum(dim=(-2, -1), keepdim=True) / s
    vyy = (w * DY * DY).sum(dim=(-2, -1), keepdim=True) / s
    vxy = (w * DX * DY).sum(dim=(-2, -1), keepdim=True) / s

    # Eigenvalues of 2x2 covariance
    d = vxx - vyy
    t = vxx + vyy
    disc = torch.sqrt((d * d) + 4 * (vxy * vxy) + eps)
    lam_max = 0.5 * (t + disc)
    lam_min = 0.5 * (t - disc).clamp_min(eps)

    smaj = torch.sqrt(lam_max + eps)        # sigma in px
    smin = torch.sqrt(lam_min + eps)
    k = 2.354820045  # FWHM/sigma
    fmaj = k * smaj
    fmin = k * smin
    axial = (fmin / (fmaj + eps))           # ≤ 1

    # 2θ encoding directly from covariance (angle-free, stable)
    denom = torch.sqrt((d * d) + 4 * (vxy * vxy) + eps)
    c2 = (d / denom).squeeze(-1).squeeze(-1)
    s2 = ((2 * vxy) / denom).squeeze(-1).squeeze(-1)

    # Pixels per (Gaussian) beam: 2π σ_maj σ_min
    px_beam = (2 * math.pi * smaj * smin).clamp_min(eps)

    feats = torch.cat([
        torch.log1p(fmaj).squeeze(-1).squeeze(-1),
        torch.log1p(fmin).squeeze(-1).squeeze(-1),
        axial.squeeze(-1).squeeze(-1),
        c2,
        s2,
        torch.log1p(px_beam).squeeze(-1).squeeze(-1),
    ], dim=1)
    return feats

# --- single encoder + per-layer heads ---
class PSFContext(nn.Module):
    """Encode PSF (centered, peak-1) once → shared embedding z, then per-layer heads."""
    def __init__(self, in_ch=1, ctx_ch=128, max_capacity=64, n_layers=4, use_feats=True, feat_dim=6):
        super().__init__()
        self.use_feats = use_feats
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, max_capacity//4, 7, stride=2, padding=3, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(max_capacity//4, max_capacity//2, 5, stride=2, padding=2, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(max_capacity//2, max_capacity, 3, stride=2, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # → (B,64,1,1)
        )
        in_proj = max_capacity + (feat_dim if use_feats else 0)
        self.shared = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_proj),
            nn.Linear(in_proj, ctx_ch),
            nn.GELU(),
        )
        # one small head per FiLM block (keeps contexts distinct with tiny cost)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(ctx_ch, ctx_ch), nn.GELU())
            for _ in range(n_layers)
        ])

    def forward(self, psf_centered: torch.Tensor):
        """
        psf_centered: (B,1,H,W), centered & peak=1.
        Returns:
          z_shared: (B, ctx_dim)
          ctx_list: list of (B, ctx_dim) per FiLM block
        """
        x = self.enc(psf_centered)          # (B,64,1,1)
        x = x.flatten(1)                    # (B,64)
        if self.use_feats:
            feats = psf_geom_features(psf_centered)  # (B,6)
            x = torch.cat([x, feats], dim=1)
        z = self.shared(x)                  # (B,ctx_dim)
        ctxs = [head(z) for head in self.heads]
        return z, ctxs

class FiLMBlock(nn.Module):
    """
    Wraps an arbitrary conv block and applies stable FiLM to its OUTPUT.

    Key ideas:
      - Normalize context (LayerNorm) -> small MLP -> (gamma, beta)
      - Normalize features (GroupNorm) before modulation
      - Softly bound gain via tanh; learnable gate blends modulation
      - No channel-shape assumptions beyond knowing the block's output channels

    Args:
      block:    nn.Module producing (B, C_out, H, W)
      feat_ch:  C_out of the wrapped block (channels AFTER the block)
      ctx_ch:   context vector size (e.g., PSFContext output dim)
      hidden:   hidden width in the ctx MLP
      max_gain: cap on |gamma| via tanh (e.g., 0.5 => at most ±50% scaling)
      num_groups: GroupNorm groups for feature normalization
      gate_init: initial gate (sigmoid(gate)) ~ how much to apply FiLM at start
    """
    def __init__(
        self,
        block: nn.Module,
        feat_ch: int,
        ctx_ch: int,
        hidden: int = 128,
        max_gain: float = 0.5,
        num_groups: int = 8,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.block = block
        self.feat_ch = feat_ch
        self.max_gain = max_gain

        # Normalize context, then predict gamma/beta
        self.ctx_norm = nn.LayerNorm(ctx_ch)
        self.ctx_mlp  = nn.Sequential(
            nn.Linear(ctx_ch, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * feat_ch)
        )

        # Normalize features before modulation
        self.feat_gn = nn.GroupNorm(num_groups=min(num_groups, feat_ch), num_channels=feat_ch)

        # Learnable gate to blend modulation safely
        self.gate = nn.Parameter(torch.tensor(gate_init))

        # Init the last linear to near-zero so γ,β start ~ 0
        nn.init.zeros_(self.ctx_mlp[-1].weight)
        nn.init.zeros_(self.ctx_mlp[-1].bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, C_in, H, W) → block → (B, feat_ch, H, W)
        ctx: (B, ctx_ch)      PSF context or similar
        """
        y = self.block(x)  # (B, C, H, W) where C == feat_ch
        B, C, H, W = y.shape

        # Context → (gamma, beta)
        ctx_n = self.ctx_norm(ctx)                 # (B, ctx_ch)
        gamma_beta = self.ctx_mlp(ctx_n)           # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=1)   # (B, C), (B, C)

        # Bound gain for stability
        gamma = torch.tanh(gamma) * self.max_gain

        # Reshape for broadcasting
        gamma = gamma.view(B, C, 1, 1)
        beta  = beta .view(B, C, 1, 1)

        # Modulate normalized features; blend via gate
        y_n   = self.feat_gn(y)
        y_mod = y_n * (1.0 + gamma) + beta

        g = torch.sigmoid(self.gate)  # scalar in (0,1)
        out = y + g * (y_mod - y_n)
        return out

class VIREO(pl.LightningModule):
    """
    A convolutional UNet for astronomical images using PSF context.

    This UNet:
      - Encodes images of shape (B, C, 600, 600) into a latent space of shape 
        (B, latent_channels, 75, 75) by downsampling with a factor of 8.
      - Uses a standard reparameterization trick.
      - Decodes the latent back to the original image resolution.
    
    The reconstruction loss is computed as MSE (assuming images are normalized 
    to [-1, 1]) plus a KL divergence term weighted by beta.
    
    Once trained, the encoder and decoder will serve as the latent space 
    representation needed for the latent diffusion model.
    """
    def __init__(self, 
                 image_size: int = 600,
                 in_channels: int = 1,
                 latent_channels: int = 4,
                 latent_scale: float = 1.,
                 ctx_ch: int = 128,
                 ctx_capacity: int = 64,
                 lr: float = 1e-4,
                 adam_eps: float = 1e-7,
                 dropout: float = 0.1,
                 plot_test: bool = True,
                 pct_start: float = 0.1,
                 laplace_weight: float = 0.01,
                 wandb_name: str = "vireo",
                 telescope: str = "ska",
                 max_capacity: int = 128,
                 recon_loss_type: str = "ms_ssim",
                 skip_drop_prob: float = 0.5,
                 skip_dropout_noise: float = 0.1,
                 add_skip_noise: bool = False,
                 zero_skips: bool = True,
                 schedule_skip_drop: bool = True,
                 dropblock_size: int = 7,
                 num_groups: int = 8,
                 dropblock_prob: float = 0.1,
                 schedule_dropblock: bool = False,
                 max_epochs: int = 100,
                 window_size: int = 0,
                 calc_mssim_every: int = 10,
                 calc_starlet_every: int = 10,
                 scale_out: float = 0.5,
                 use_inception: bool = True,
                 num_heads: int = 1,
                 add_final_skip: bool = True,
                 grow_scale: float = 1.,
                 double_weights: bool = True,
                 which_device: str = "mps",
                 intermediate_attention: bool = True,
                 final_attention: bool = True,
                 learn_diff: bool = True,
                 final_activation: str = "asinh",
                 asinh_k: float = 10.,
                 softclip_a: float = 0.1,
                 grad_clip: float = 1.,
                 n_patches: int = 5,
                 psf_min: int = 2,
                 psf_max: int = 8,
                 dc_blend: float = 0.5,
                 dc_lam: float = 0.1,
                 dc_unroll_steps: int = 0,
                 w_beam: float = 1.,
                 w_flux: float = 0.,
                 ssim_weight: float = 0.75,
                 enforce_nonneg: bool = False,
                 recon_weight: float = 0.5,
                 use_denoise_loss: bool = True,
                 fwd_weight: float = 0.25,
                 x_dc_weight: float = 1.,
                 w_spectral: float = 0.0,
                 w_low: float = 0.0,
                 w_high: float = 0.0,
                 w_beam_img: float = 0.,
                 w_oob: float = 0.0,
                 w_blur: float = 0.1,
                 w_starlet: float = 0.5,
                 w_psd: float = 0.0,
                 schedule_loss: bool = True,
                 ramp_epochs: int = 10,
                 add_film: bool = True,
                 dc_T: int = 2,
                 wiener_init: int = 0.5,
                 use_self_ensemble: bool = False,
                 transform: bool = True,
                 norm_psf: bool = False,
                 use_dc: bool = False,
                 use_white_dc: bool = False,
                 use_psf: bool = True,
                 sum_norm_psf: bool = True,
                 **kwargs,
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.image_size = image_size
        self.telescope = telescope
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.latent_scale = latent_scale
        self.ctx_ch = ctx_ch
        self.ctx_capacity = ctx_capacity
        self.lr = lr
        self.adam_eps = adam_eps
        self.pct_start = pct_start
        self.dropout = dropout
        self.model_type = "vireo" if use_psf else "psf_ignorant"
        self.wandb_name = wandb_name
        self.max_capacity = max_capacity
        self.n_patches = n_patches
        self.recon_loss_type = recon_loss_type
        self.skip_drop_prob = skip_drop_prob
        self.skip_dropout_noise = skip_dropout_noise
        self.add_skip_noise = add_skip_noise
        self.zero_skips = zero_skips
        self.schedule_skip_drop = schedule_skip_drop
        self.dropblock_size = dropblock_size
        self.num_groups = num_groups
        self.dropblock_prob = dropblock_prob
        self.max_epochs = max_epochs
        self.window_size = window_size
        self.laplace_weight = laplace_weight
        self.calc_mssim_every = calc_mssim_every
        self.calc_starlet_every = calc_starlet_every
        self.final_activation = final_activation
        self.scale_out = scale_out
        self.use_inception = use_inception
        self.num_heads = num_heads
        self.add_final_skip = add_final_skip
        self.grow_scale = grow_scale
        self.double_weights = double_weights
        self.intermediate_attention = intermediate_attention
        self.final_attention = final_attention
        self.learn_diff = learn_diff
        self.asinh_k = asinh_k
        self.softclip_a = softclip_a
        self.psf_min = psf_min
        self.psf_max = psf_max
        self.grad_clip = grad_clip
        self.dc_blend = dc_blend
        self.dc_lam = dc_lam
        self.dc_unroll_steps = dc_unroll_steps
        self.w_beam = w_beam
        self.w_flux = w_flux
        self.ssim_weight = ssim_weight
        self.recon_weight = recon_weight if use_dc else 0.
        self.enforce_nonneg = enforce_nonneg
        self.use_denoise_loss = use_denoise_loss
        self.fwd_weight = fwd_weight if use_psf else 0.
        self.x_dc_weight = x_dc_weight
        self.w_spectral = w_spectral
        self.w_low = w_low
        self.w_high = w_high
        self.w_beam_img = w_beam_img if use_psf else 0.
        self.w_oob = w_oob if use_psf else 0.
        self.w_blur = w_blur if use_psf else 0.
        self.w_psd = w_psd
        self.w_starlet = w_starlet
        self.schedule_loss = schedule_loss
        self.ramp_epochs = ramp_epochs
        self.add_film = add_film if use_psf else False
        self.dc_T = dc_T
        self.wiener_init = wiener_init
        self.use_self_ensemble = use_self_ensemble
        self.transform = transform
        self.norm_psf = norm_psf
        self.use_dc = use_dc
        self.use_white_dc = use_white_dc
        self.use_psf = use_psf
        self.sum_norm_psf = sum_norm_psf
        
        self.save_hyperparameters()
        
        if schedule_skip_drop:
            self.register_buffer("skip_drop_schedule", torch.linspace(self.skip_drop_prob, 0.0, steps=self.max_epochs))
        
        if schedule_dropblock:
            self.register_buffer("dropblock_schedule", torch.linspace(self.dropblock_prob, 0.0, steps=self.max_epochs))
        
        self.plot_on_batch = 0
        self.plot_val = True
        self.plotted = False
        
        self.mssim = multiscale_structural_similarity_index_measure
        self.ssim = structural_similarity_index_measure
        
        self.plotted = False
        self.plot_test = plot_test
        
        self.scaler = GradScaler()
        
        self.example_input_array = (
            torch.zeros(1, in_channels, image_size, image_size),
            torch.zeros(1, in_channels, image_size, image_size),
        )

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, max_capacity//4, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            ResidualBlock(max_capacity//4, max_capacity//4, kernel_size=3, dropout=dropout, 
                          dropblock_prob=dropblock_prob, dropblock_size=dropblock_size) if not self.use_inception else\
            InceptionResBlock(max_capacity//4, max_capacity//4, dropout=dropout,
                              dropblock_prob=dropblock_prob, dropblock_size=dropblock_size)
        )  # 600 -> 300
        self.down2 = nn.Sequential(
            nn.Conv2d(max_capacity//4, max_capacity//2, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            ResidualBlock(max_capacity//2, max_capacity//2, kernel_size=3, dropout=dropout,
                          dropblock_prob=dropblock_prob, dropblock_size=dropblock_size - 2, dilation=2),
        )  # 300 -> 150
        self.down3 = nn.Sequential(
            nn.Conv2d(max_capacity//2, max_capacity, kernel_size=4, stride=2, padding=2, padding_mode="reflect"),
            ResidualBlock(max_capacity, max_capacity, kernel_size=3, dropout=dropout, 
                          dropblock_prob=dropblock_prob, dropblock_size=dropblock_size - 4)
        )  # 150 -> 76

        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(max_capacity, max_capacity, kernel_size=3, dropout=dropout, 
                          dropblock_prob=0.) if not self.use_inception else\
            InceptionResBlock(max_capacity, max_capacity, dropout=dropout,
                              dropblock_prob=0.),
            nn.GroupNorm(num_groups=min(8, max_capacity), 
                                 num_channels=max_capacity),
            SelfAttention(max_capacity, window_size=self.window_size) if num_heads == 1 else\
            MultiHeadSelfAttention(max_capacity, num_heads=num_heads)
        )  # 76 -> 76
        
        # Decoder
        # Stage1: 76→150 px
        self.up1_upsample, self.up1_fuse = self.make_decoder_stage(
            in_channels=max_capacity,
            out_channels=max_capacity,
            skip_channels=max_capacity//2,
            window_size=0,
            dropout=dropout,
            dropblock_prob=dropblock_prob,
            dropblock_size=dropblock_size-4,
            dilation=1,
            size=150,
            )

        # Stage2: 150→300 px
        self.up2_upsample, self.up2_fuse = self.make_decoder_stage(
            in_channels=max_capacity,
            out_channels=max_capacity//2,
            skip_channels=max_capacity//4,
            window_size=0,
            dropout=dropout,
            dropblock_prob=dropblock_prob,
            dropblock_size=dropblock_size-2,
            dilation=2,
        )

        # Stage3: 300→600 px
        self.up3_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(max_capacity//2, max_capacity//4, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GELU(),
        )
        self.up3_fuse = InceptionResBlock(max_capacity//4, 
                              max_capacity//4, dropout=dropout,
                              dropblock_prob=0., 
                              dropblock_size=dropblock_size) if not self.final_attention else\
            nn.Sequential(
                nn.GroupNorm(num_groups=min(8, max_capacity//4), num_channels=max_capacity//4),
                SlidingWindowMHSA(max_capacity//4, num_heads=1, patch_size=8),
                nn.GroupNorm(num_groups=min(8, max_capacity//4), num_channels=max_capacity//4),
                InceptionResBlock(max_capacity//4, 
                              max_capacity//4, dropout=dropout,
                              dropblock_prob=0., 
                              dropblock_size=dropblock_size) 
            )
            
        # change to FiLM blocks
        if self.add_film:
            self.up1_fuse = FiLMBlock(self.up1_fuse, feat_ch=self.max_capacity,      ctx_ch=self.ctx_ch)
            self.up2_fuse = FiLMBlock(self.up2_fuse, feat_ch=self.max_capacity//2,   ctx_ch=self.ctx_ch)
            self.up3_fuse = FiLMBlock(self.up3_fuse, feat_ch=self.max_capacity//4,   ctx_ch=self.ctx_ch)
            self.bottleneck = FiLMBlock(self.bottleneck, feat_ch=self.max_capacity, ctx_ch=self.ctx_ch)
            
            self.psf_ctx = PSFContext(in_ch=1, ctx_ch=self.ctx_ch, max_capacity=self.ctx_capacity, n_layers=3)
            
            self.bn_head = nn.Sequential(
                nn.LayerNorm(self.ctx_ch),
                nn.Linear(self.ctx_ch, self.ctx_ch),
                nn.GELU(),
                nn.Linear(self.ctx_ch, self.ctx_ch)
            )
            
        self.alpha_dc_1 = nn.Parameter(torch.tensor(0.3))
        self.alpha_dc_2 = nn.Parameter(torch.tensor(0.5))
        
        self.head1 = nn.Conv2d(max_capacity, 1, kernel_size=1)   # if up1_fuse outputs max_capacity channels
        self.head2 = nn.Conv2d(max_capacity//2, 1, kernel_size=1)   # if up2_fuse outputs max_capacity//2 channels

        self.edge_smoother = nn.Sequential(
            nn.Conv2d(
                in_channels=max_capacity//4,
                out_channels=max_capacity//4,
                kernel_size=3,
                padding=1,
                padding_mode='reflect',
                groups=max_capacity//4         # <-- depthwise!
            ),
            nn.Conv2d(max_capacity//4, max_capacity//4, kernel_size=1),
            nn.GELU(),
        )
        
        # Final 1x1 conv
        self.up4 = nn.Sequential(
            nn.Conv2d(max_capacity//4, max_capacity//4, kernel_size=1),
            nn.GroupNorm(min(8, max_capacity//4), max_capacity//4),
            nn.GELU(),
            nn.Conv2d(max_capacity//4, in_channels, kernel_size=1),
        )
        
        if self.final_activation == "asinh":
            self.final_act = AsinhHead(self.asinh_k)
        elif self.final_activation == "softclip":
            self.final_act = SoftClipHead(self.softclip_a)
        elif self.final_activation == "tanh":
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()
        
        if self.scale_out > 0.:
            self.out_scale = nn.Parameter(torch.tensor(self.scale_out))

        self.apply(init_weights)
        
        self.se = SelfEnsemble()
        
        for name, param in self.named_parameters():
            if param.numel() == 0:
                print("Empty parameter:", name, param.shape)
                
        if self.use_denoise_loss:
            # oob, blur, beam_img, psd, fwd
            self.denoise_loss = DenoiseLoss(w_main=self.x_dc_weight, 
                                            w_ssim=self.ssim_weight, 
                                            w_aux=self.recon_weight, 
                                            w_fwd=self.fwd_weight,
                                            w_spectral=self.w_spectral,
                                            w_beam_img=self.w_beam_img,
                                            w_oob=self.w_oob,
                                            w_starlet=self.w_starlet,
                                            calc_msssim_every=self.calc_mssim_every,
                                            calc_starlet_every=self.calc_starlet_every,
                                            schedule_loss=self.schedule_loss,
                                            ramp_epochs=self.ramp_epochs,
                                            w_low=self.w_low,
                                            w_high=self.w_high,
                                            w_blur=self.w_blur,
                                            w_psd=self.w_psd,
                                            )
            
            
        if self.transform:
            self._make_aug()
    
                
    def make_decoder_stage(self, in_channels, skip_channels, window_size, dropout, 
                           dropblock_prob, dropblock_size, dilation, out_channels, size=None):
        # always want: Upsample → Conv → GELU
        up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if size is None\
                        else nn.Upsample(size=(size, size), mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="reflect"),
            nn.GELU(),
        )
        tot_channels = out_channels + skip_channels
        # build the “fuse” module: either [Attn → ResBlock] or just ResBlock
        if window_size > 0:
            fuse = nn.Sequential(
                nn.GroupNorm(num_groups=min(8, tot_channels), num_channels=tot_channels),
                SelfAttention(tot_channels, window_size=window_size),
                nn.GroupNorm(num_groups=min(8, tot_channels), num_channels=tot_channels),
                ResidualBlock(
                    in_channels=tot_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    dropblock_prob=dropblock_prob,
                    dropblock_size=dropblock_size,
                    dilation=dilation,
                )
            )
        else:
            fuse = ResidualBlock(
                in_channels=tot_channels,
                out_channels=out_channels,
                dropout=dropout,
                dropblock_prob=dropblock_prob,
                dropblock_size=dropblock_size,
                dilation=dilation,
            )
        return up, fuse
        
    def encode(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        e1 = self.down1(x)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        
        # build padded psf for context (centered, unit-sum)
        B, _, H, W = x.shape
        
        b  = self.bottleneck(e3, ctx) if self.add_film else self.bottleneck(e3)
        return e1, e2, e3, b
        
    def decode(self, e1, e2, e3, b, x, psf, ctxs):
        """
        Multi-scale decode with Wiener DC applied at each scale in the image domain.
        e1, e2, e3: encoder features at different scales
        b: bottleneck features
        x: input dirty image (B,1,H,W)
        otf: OTF in Fourier space (B,1,H,W//2+1)
        ctx: Context features from PSF (B,ctx_ch)
        """
        dirty = x
        B, _, H, W = x.shape
        
        h2, w2 = H // 4, W // 4   # stage1 spatial size
        h3, w3 = H // 2, W // 2   # stage2 spatial size
        
        dirty_2 = F.interpolate(dirty, size=(h2, w2), mode='area')
        dirty_3 = F.interpolate(dirty, size=(h3, w3), mode='area')

        # ---------------------------
        # Stage 1
        # ---------------------------
        d1 = self.up1_upsample(b)
        d1 = torch.cat([d1, self._maybe_drop(e2)], dim=1)
        d1 = self.up1_fuse(d1, ctxs[0]) if self.add_film else self.up1_fuse(d1)

        # Project to image, Wiener DC, inject back into features
        d1_img = self.head1(d1)  # (B,1,H1,W1)
        if self.use_dc:
            d1_img_dc = self.dc_identity_flipped(d1_img, dirty_2, self.dc_lam)
            d1_dc = d1 + self.alpha_dc_1 * (d1_img_dc - d1_img)
        else:
            d1_dc = d1

        # ---------------------------
        # Stage 2
        # ---------------------------
        d2 = self.up2_upsample(d1_dc)
        d2 = torch.cat([d2, self._maybe_drop(e1)], dim=1)
        d2 = self.up2_fuse(d2, ctxs[1]) if self.add_film else self.up2_fuse(d2)

        # Project to image, Wiener DC, inject back into features
        d2_img = self.head2(d2)  # (B,1,H2,W2)
        if self.use_dc:
            d2_img_dc = self.dc_identity_flipped(d2_img, dirty_3, self.dc_lam)
            d2_dc = d2 + self.alpha_dc_2 * (d2_img_dc - d2_img)
        else:
            d2_dc = d2

        # ---------------------------
        # Stage 3
        # ---------------------------
        d3 = self.up3_upsample(d2_dc)
        d3 = self.up3_fuse(d3, ctxs[2]) if self.add_film else self.up3_fuse(d3)

        # Edge enhancement
        edge = K.filters.laplacian(d3, kernel_size=3)
        d3 = d3 + self.laplace_weight * edge
        d3 = nn.GELU()(d3)
        d3 = d3 + self.edge_smoother(edge)
        
        # ---------------------------
        # Final output
        # ---------------------------
        delta = self.up4(d3)
        if self.scale_out > 0.:
            delta = self.final_act(delta * self.out_scale)
        else:
            delta = self.final_act(delta)

        if self.learn_diff:
            recon = x + delta
        else:
            recon = delta
            
        if self.enforce_nonneg:
            recon = self._softplus_nonneg(recon)
        
        if self.use_white_dc:
            otf = self._psf_to_otf(psf)
            x_dc = self.dc_identity_whitened(recon, dirty, otf)
        elif self.use_dc:
            x_dc = self.dc_identity_flipped(recon, dirty, self.dc_lam)
        else:
            x_dc = recon

        return {"out": recon, "out_dc": x_dc}
    
    def forward(self, x: torch.Tensor, psf: torch.Tensor,) -> torch.Tensor:
        if self.add_film:
            z_shared, ctxs = self.psf_ctx(psf)
            z_shared = self.bn_head(z_shared)
        else:
            z_shared = None
            ctxs = None

        e1, e2, e3, b = self.encode(x, z_shared)
        return self.decode(e1, e2, e3, b, x, psf, ctxs)
    
    def _downscale_otf(self, otf_r, target_hw):
        _, _, H, Wr = otf_r.shape
        h, w = target_hw
        wr = w // 2 + 1
        top = (H - h) // 2
        # For rfft, crop sym in vertical, truncate in freq domain horizontally
        return otf_r[:, :, top:top+h, :wr]
    
    def _make_aug(self):
        vert_p = np.random.uniform(0, 1.)
        horiz_p = np.random.uniform(0, 1.)
        rot_deg = np.random.uniform(0, 180.)
        rot_p = np.random.uniform(0, 1.)
        
        self.aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=horiz_p),
            K.augmentation.RandomVerticalFlip(p=vert_p),
            K.augmentation.RandomRotation(degrees=rot_deg, p=rot_p),
            same_on_batch=False,  # different per sample in a batch
            data_keys=["input", "input", "input"],
        )

    def _maybe_drop(self, skip: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return skip
        skip_prob = self.skip_drop_prob if not self.schedule_skip_drop else self.skip_drop_schedule[self.current_epoch]
        if self.add_skip_noise and torch.rand(1).item() < skip_prob:
            return torch.randn_like(skip) * self.skip_dropout_noise
        if self.zero_skips and torch.rand(1).item() < skip_prob:
            return torch.zeros_like(skip)
        return skip

    def _irfft2(self, X, size_hw):
        return torch.fft.irfft2(X, s=size_hw, norm="ortho")
    
    def _rfft2(self, x):
        return torch.fft.rfft2(x, norm="ortho")

    def _psf_to_otf(
        self,
        psf,                         # (B,1,H,W) or (B,H,W) or (H,W), real
        out_hw=None,                 # (H,W) of the target image; if None, use psf size
        psf_is_centered: bool = True,# True = peak at center; False = peak at [0,0]
        reassert_peak: bool = True,  # make center/[0,0] exactly +1 before FFT
        eps: float = 1e-12,
        ):
        import torch, torch.nn.functional as F
        psf = torch.as_tensor(psf)
        if psf.ndim == 2:
            psf = psf.unsqueeze(0).unsqueeze(0)     # (1,1,H,W)
        elif psf.ndim == 3:
            psf = psf.unsqueeze(1)                  # (B,1,H,W)
        B, C, h, w = psf.shape

        # optional crop/pad to out size
        if out_hw is not None:
            H, W = int(out_hw[0]), int(out_hw[1])
            if H < h or W < w:
                top = (h - H)//2; left = (w - W)//2
                psf = psf[..., top:top+H, left:left+W]
                h, w = H, W
            pad_h = (H - h)//2; pad_w = (W - w)//2
            psf = F.pad(psf, (pad_w, W-w-pad_w, pad_h, H-h-pad_h))
        else:
            H, W = h, w

        # bring to centered contract if needed for normalization
        if not psf_is_centered:
            psf = torch.fft.fftshift(psf, dim=(-2, -1))

        # robustly re-assert peak=+1 at the center (prevents tiny-denominator blowups)
        if reassert_peak:
            cy, cx = H//2, W//2
            center = psf[..., cy, cx]
            sign = torch.sign(center); sign[sign==0] = 1.0
            psf = psf * sign.unsqueeze(-1).unsqueeze(-1)
            den = psf[..., cy, cx].abs().clamp_min(1e-12)
            psf = psf / den.unsqueeze(-1).unsqueeze(-1)
            
        psf_sum1 = psf / torch.sum(psf, dim=(-2, -1), keepdim=True)

        # single ifftshift before FFT (because we keep PSFs centered in the model)
        psf_sum1 = torch.fft.ifftshift(psf_sum1, dim=(-2, -1))

        # OTF in the same normalization used elsewhere
        P = torch.fft.ifftshift(psf_sum1, dim=(-2,-1))        # remove center offset
        H = torch.fft.rfft2(P, norm="backward")                # complex OTF

        # normalize DC no matter the FFT norm:
        dc = H[..., :1, :1]                       # keep dims, complex
        mag2 = (dc.real*dc.real + dc.imag*dc.imag).clamp_min(eps)  # real, safe
        inv_dc = dc.conj() / mag2                 # stable complex reciprocal

        H_unitdc = H * inv_dc                     # now DC ≈ 1+0j
        return H_unitdc
    
    # B. reassert peak=1 AFTER pad/crop (torch version)
    def _recenter_peak1(self, psf: torch.Tensor, contract: str = "center", eps=1e-12):
        """
        contract: "center" (peak at [H//2,W//2]) or "ifft" (peak at [0,0]).
        """
        if psf.dim() == 2: psf = psf.unsqueeze(0).unsqueeze(0)
        elif psf.dim() == 3: psf = psf.unsqueeze(1)
        B,C,H,W = psf.shape
        x = psf.clone()

        flat = x.abs().reshape(B*C, -1)
        idx = flat.argmax(dim=-1)
        iy = (idx // W).view(B,C); ix = (idx % W).view(B,C)

        ty, tx = (H//2, W//2) if contract=="center" else (0, 0)
        for b in range(B):
            for c in range(C):
                dy = int(iy[b,c].item()) - ty
                dx = int(ix[b,c].item()) - tx
                if dy or dx:
                    x[b,c] = torch.roll(x[b,c], shifts=(-dy, -dx), dims=(-2,-1))

        center = x[..., ty, tx]
        sign = torch.sign(center); sign[sign==0] = 1.0
        x = x * sign.unsqueeze(-1).unsqueeze(-1)
        den = x[..., ty, tx].abs().clamp_min(eps)
        x = x / den.unsqueeze(-1).unsqueeze(-1)
        return x if psf.dim()==4 else x.squeeze(1)
    
    def _pad_psf_to(self, psf: torch.Tensor, hw, norm_mode: str = "peak"):
        # Expect psf peak at the CENTER already (centered contract)
        if psf.dim() == 3:  # (B,H,W) -> (B,1,H,W)
            psf = psf.unsqueeze(1)
        assert psf.dim() == 4
        B, C, h, w = psf.shape
        H, W = int(hw[0]), int(hw[1])

        # center-crop then symmetric pad
        if H < h or W < w:
            top = (h - H) // 2; left = (w - W) // 2
            psf = psf[..., top:top+H, left:left+W]
            h, w = H, W
        pad_h = (H - h) // 2; pad_w = (W - w) // 2
        psf_big = F.pad(psf, (pad_w, W-w-pad_w, pad_h, H-h-pad_h))

        # normalization AFTER pad/crop
        if norm_mode == "peak":
            cy, cx = H//2, W//2
            center = psf_big[..., cy, cx]
            sign = torch.sign(center); sign[sign==0] = 1.0
            psf_big = psf_big * sign.unsqueeze(-1).unsqueeze(-1)
            den = psf_big[..., cy, cx].abs().clamp_min(1e-12)
            psf_big = psf_big / den.unsqueeze(-1).unsqueeze(-1)

        elif norm_mode == "sum":
            s = psf_big.sum(dim=(-2,-1), keepdim=True).clamp_min(1e-12)
            psf_big = psf_big / s
        elif norm_mode == "none":
            pass
        else:
            raise ValueError("norm_mode must be 'peak', 'sum', or 'none'.")
        return psf_big

    @staticmethod
    def _softplus_nonneg(x, beta=1.0, threshold=20.0):
        return F.softplus(x, beta=beta, threshold=threshold)

    def dc_identity(self, pred_img, dirty_img, lam=0.1):
        # (dirty + lam*pred) / (1+lam)
        return (dirty_img + lam*pred_img) / (1.0 + lam)
    
    def dc_identity_flipped(self, pred_img, dirty_img, lam=0.1):
        # (lam*dirty + pred) / (1+lam)
        return (lam * dirty_img + pred_img) / (1.0 + lam)
    
    def dc_diff(self, pred_img, dirty_img, tau=0.05):
        # tau=0 ⇒ keep recon; tau=1 ⇒ replace by dirty
        return pred_img + tau * (dirty_img - pred_img)
    
    def dc_identity_whitened(self, recon, dirty, otf_restoring, lam=0.1, eps=1e-2, use_ortho=True):
        norm = "ortho" if use_ortho else "backward"
        Rk = otf_restoring
        # whitening filter ~ R* / (|R|^2 + eps^2)
        eps_rel = eps * Rk.abs().amax(dim=(-2,-1), keepdim=True).clamp_min(1e-8)
        Wk = torch.conj(Rk) / (Rk.abs()**2 + eps_rel**2)

        rk  = torch.fft.rfft2(recon, norm=norm)
        yk  = torch.fft.rfft2(dirty, norm=norm)
        rk_w, yk_w = Wk*rk, Wk*yk

        xk_w = (yk_w + self.dc_lam * rk_w) / (1.0 + self.dc_lam)      # identity DC in whitened space
        xk   = xk_w * Rk                               # unwhiten back
        x    = torch.fft.irfft2(xk, s=recon.shape[-2:], norm=norm).real
        return x
    
    def dc_fusion(self, x_net, y_dirty, H, lam=0.5):
        # H: OTF with DC normalized to 1, shape broadcastable to rfft2(x_net)
        X = torch.fft.rfft2(x_net, norm="ortho")
        Y = torch.fft.rfft2(y_dirty, norm="ortho")
        num = H.conj() * Y + lam * X
        den = (H.real**2 + H.imag**2) + lam
        Xdc = num / den.clamp_min(1e-12)
        return torch.fft.irfft2(Xdc, s=x_net.shape[-2:], norm="ortho")
        
    @torch.no_grad()
    def beam_core_from_psf(self, psf_centered: torch.Tensor,
                                frac: float = 0.5,
                                eps: float = 1e-12):
        """
        Estimate the beam core from a *centered*, peak-1 PSF.
        psf_centered: (B,1,H,W) or (B,H,W) or (H,W), with peak at (H//2, W//2)
        frac: keep pixels where PSF >= frac * peak (e.g., 0.3–0.7)

        Returns dict of tensors (shape [B,1] except angles are [B,1]):
        fwhm_major_px, fwhm_minor_px, bpa_deg (east of north), px_per_beam
        """
        p = psf_centered
        if p.dim() == 2: p = p.unsqueeze(0).unsqueeze(0)
        elif p.dim() == 3: p = p.unsqueeze(1)
        B, C, H, W = p.shape
        assert C == 1, "expect single-channel PSF"

        device = p.device
        cy, cx = H//2, W//2

        # coords centered at peak
        yy = torch.arange(H, device=device).float() - cy
        xx = torch.arange(W, device=device).float() - cx
        YY, XX = torch.meshgrid(yy, xx, indexing="ij")         # (H,W)

        peak = p[..., cy, cx].clamp_min(eps)                   # (B,1)
        core_mask = (p >= (frac * peak).unsqueeze(-1).unsqueeze(-1)).float()

        # non-negative weights from core only (ignore sidelobes)
        w = torch.clamp(p, min=0.0) * core_mask                # (B,1,H,W)
        s = w.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)   # (B,1,1,1)
        # moments
        mx = (w * XX).sum(dim=(-1, -2), keepdim=True) / s
        my = (w * YY).sum(dim=(-1, -2), keepdim=True) / s
        DX = XX - mx; DY = YY - my
        vxx = (w * DX * DX).sum(dim=(-1, -2), keepdim=True) / s
        vyy = (w * DY * DY).sum(dim=(-1, -2), keepdim=True) / s
        vxy = (w * DX * DY).sum(dim=(-1, -2), keepdim=True) / s

        # eigendecomp of 2x2 covariance per batch (closed form)
        t = vxx + vyy
        d = vxx - vyy
        r = torch.sqrt((0.5*d)**2 + vxy**2 + eps)
        lam_max = 0.5*(t + 2*r).clamp_min(eps)
        lam_min = 0.5*(t - 2*r).clamp_min(eps)
        sigma_major = torch.sqrt(lam_max)
        sigma_minor = torch.sqrt(lam_min)
        fwhm_major_px = 2.354820045 * sigma_major
        fwhm_minor_px = 2.354820045 * sigma_minor

        # angle wrt +x (radians), convert to East-of-North (deg)
        theta_x = 0.5 * torch.atan2(2*vxy, d + eps)           # [-pi/2, pi/2]
        bpa_deg = (90.0 - (theta_x * 180.0 / math.pi)) % 180.0

        px_per_beam = (2*math.pi * sigma_major * sigma_minor) # (B,1,1,1)

        return dict(
            fwhm_major_px=fwhm_major_px.squeeze(-1).squeeze(-1),  # (B,1)
            fwhm_minor_px=fwhm_minor_px.squeeze(-1).squeeze(-1),
            bpa_deg=bpa_deg.squeeze(-1).squeeze(-1),
            px_per_beam=px_per_beam.squeeze(-1).squeeze(-1),
        )

    def process_batch(self, batch, batch_idx, phase: str = "train", return_recon: bool = False):
        
        t = {}
        x, x_dirty, psf = batch["clean"], batch["dirty"], batch["psf"]
        if psf.shape != x.shape:
            print("Padded PSF")
            psf = self._pad_psf_to(psf, (x.shape[-2], x.shape[-1]))

        if self.transform and phase == "train":
            with timer("augment", t):
                x, x_dirty, psf = self.aug(x, x_dirty, psf)
                
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        if len(x_dirty.shape) == 3:
            x_dirty = x_dirty.unsqueeze(1)
        
        if phase == "train":
            optimizer = self.optimizers()

        with timer("forward", t):
            out_dict = self(x_dirty, psf,)
        with timer("loss", t):
            if self.sum_norm_psf:
                psf /= torch.sum(psf, dim=(-2, -1), keepdim=True)
            loss, deep_losses = self.denoise_loss(out_dict["out"], out_dict["out_dc"], x, x_dirty, 
                                                    psf=psf, batch_idx=batch_idx, phase=phase)

            
        if phase == "train":
            
            with timer("backward+step", t):
                optimizer.zero_grad()
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.clip_gradients(optimizer, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    try:
                        # loss.backward()
                        self.manual_backward(loss)
                    except Exception as e:
                        print(e)
                        print(loss)
                        breakpoint()
                    self.clip_gradients(optimizer, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")
                    optimizer.step()
            with timer("lr_step", t):
                scheduler = self.lr_schedulers()[0]
                scheduler.step()
                
        self.log(f"{phase}/loss", loss.detach(), prog_bar=True)
        for key, value in deep_losses.items():
            self.log(f"{phase}/{key}", value, prog_bar=False)
            if key == "base_loss":
                continue
            self.log(f"{phase}/scaled_{key}", 
                     value * self.denoise_loss.current_weights[self.denoise_loss.metric_to_weight[key]], 
                     prog_bar=False)
        
        if batch_idx % 2 == 0:
            self.log_dict({f"time/{k}": v for k, v in t.items()}, prog_bar=False, on_step=True, on_epoch=False)
            
        if self.sum_norm_psf:
            psf = psf / psf.amax(dim=(-1,-2), keepdim=True)
        
        return loss if not return_recon else {"loss": loss, 
                                              "x": x, 
                                              "x_recon": out_dict["out"], 
                                              "x_dirty": x_dirty,
                                              "x_dc": out_dict["out_dc"],
                                              'psf': psf,
                                              }
        
    def training_step(self, batch, batch_idx):
        
        _ = self.process_batch(batch, batch_idx, phase="train")
        return None
    
    def validation_step(self, batch, batch_idx):
        self.log("global_step", self.global_step)
        results = self.process_batch(batch, batch_idx, phase="val", return_recon=True)
        loss = results["loss"]
        if self.current_epoch % 2 == 0 and self.plot_val and not self.plotted and batch_idx == self.plot_on_batch:
            self.plot_batch_results(results, phase="val")
                
        return loss
    
    def test_step(self, batch, batch_idx):
        results = self.process_batch(batch, batch_idx, phase="test", return_recon=True)
        loss = results["loss"]
        if self.plot_test and batch_idx % 2 == 0:
            self.plot_batch_results(results, phase="test")
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=self.adam_eps)

        # --- schedule: warmup + cosine ---
        total_steps = self.trainer.estimated_stepping_batches
        warmup_frac = 0.05          # 5% warmup
        warmup_steps = max(1, int(total_steps * warmup_frac))
        after_steps  = max(1, total_steps - warmup_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,      # start at 0.1% of self.lr
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=after_steps,
            eta_min=self.lr * 0.05  # floor at 5% of base LR
        )
        sched = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        # --- safety net: reduce LR if val stalls ---
        plateau = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-3,
            cooldown=0,
            min_lr=self.lr * 1e-3,
        )
        return [optimizer], [sched, plateau]
    
    def scale_learning_rate(self):
        for pg in self.optimizers().param_groups:
            if "lr" in pg:
                pg["lr"] = pg["lr"] * self.loss_scale
    
    def plot_batch_results(self, results, phase: str = "val"):
        
        x = results["x"]
        x_recon = results["x_recon"]
        x_dirty = results["x_dirty"]
        psf = results['psf']
        x_dc = results["x_dc"]
        b = 0
        f_dirty, psd_dirty = self.radial_psd(x_dirty[b:b+1])
        f_clean, psd_clean = self.radial_psd(x[b:b+1])
        f_recon,  psd_recon  = self.radial_psd(x_dc[b:b+1])
        f_otf, otf_mag = self.radial_otf_mag(psf[b:b+1])
        
        self.plot_dirty_predictions(x, x_dirty, x_recon, x_dc, psf,
                                    phase=phase)
        self.plot_pred_psd(psd_clean, psd_dirty, psd_recon, otf_mag,
                            f_clean, f_dirty, f_recon, f_otf,
                            phase=phase)

        if self.use_self_ensemble:
            with torch.no_grad():
                out_tta = self.se.tta_self_ensemble(
                    self,
                    dirty=x_dirty,          # (B,1,H,W)
                    psf=psf,                # (B,1,h,w) centered + normalized to peak=1
                    use_8=False,            # try False first (x4). Switch to True for maximum squeeze.
                    data_range_head="out_dc",
                )
            x_dc_tta = out_tta["out_dc"]
            self.plot_dirty_predictions(x, x_dirty, x_dc, x_dc_tta, psf, 
                                        phase=f"{phase}_tta",
                                        using_tta=True)
            
            b = 0
            f_dc_tta, psd_dc_tta = self.radial_psd(x_dc_tta[b:b+1])  
            self.plot_pred_psd(psd_clean, psd_recon, psd_dc_tta, otf_mag,
                                f_clean, f_recon, f_dc_tta, f_otf,
                                phase=f"{phase}_tta",
                                using_tta=True)
    
    def plot_dirty_predictions(self, x, x_dirty, x_recon, x_dc, psf,
                               phase: str = "val", using_tta: bool = False):
        try:
            fig, axs = plt.subplots(ncols=5, figsize=(28., 6.))
            axs = axs.flat
            ax1 = axs[0]
            ax2 = axs[1]
            ax3 = axs[2]
            ax4 = axs[3]
            ax5 = axs[4]

            img = ax1.imshow(x.cpu().numpy()[0].squeeze(), 
                    cmap="magma",
                    origin="lower",
                    # vmin=-1, vmax=1,
                    )
            img2 = ax2.imshow(x_dirty.cpu().numpy()[0].squeeze(), 
                    cmap="magma",
                    origin="lower",
                    # vmin=-1, vmax=1,
                    )
            img3 = ax3.imshow(x_recon.cpu().numpy()[0].squeeze(), 
                    cmap="magma",
                    origin="lower",
                    # vmin=-1, vmax=1,
                    )
            img4 = ax4.imshow(x_dc.cpu().numpy()[0].squeeze(), 
                    cmap="magma",
                    origin="lower",
                    # vmin=-1, vmax=1,
                    )
            img5 = ax5.imshow(psf.cpu().numpy()[0].squeeze(), 
                    cmap="magma",
                    origin="lower",
                    # vmin=-1, vmax=1,
                    )
            plt.colorbar(img, ax=ax1, fraction=0.045)
            plt.colorbar(img2, ax=ax2, fraction=0.045)
            plt.colorbar(img3, ax=ax3, fraction=0.045)
            plt.colorbar(img4, ax=ax4, fraction=0.045)
            plt.colorbar(img5, ax=ax5, fraction=0.045)
            ax1.set_title("Clean")
            ax2.set_title("Dirty")
            ax3.set_title("Recreated" if not using_tta else "DC")
            ax4.set_title("DC" if not using_tta else "DC (TTA)")
            ax5.set_title("PSF")

            image = wandb.Image(fig, caption=f"{phase} Prediction: {self.wandb_name}")
            wandb.log({f"{phase}_prediction": image})
            plt.close()
            self.plotted = True
        except Exception as e:
            print(f"Failed to plot {phase} prediction: {e}")
            
    def plot_pred_psd(self, psd_clean, psd_dirty, psd_recon, otf_mag,
                      f_clean, f_dirty, f_recon, f_otf,
                      phase: str = "val", using_tta: bool = False):
        try:
            fig = plt.figure(figsize=(8., 6.))
            plt.plot(f_clean, psd_clean[0], c="k", lw=3, label="Clean")
            plt.plot(f_dirty, psd_dirty[0], c="firebrick", lw=3, label="Dirty" if not using_tta else "DC")
            plt.plot(f_recon, psd_recon[0], c="steelblue", lw=3, label="Recreated" if not using_tta else "DC (TTA)")
            plt.plot(f_otf, otf_mag[0] ** 2, lw=3, label=r"$|R|^{2}$ (beam)", ls='--', c="gray")
            plt.title("PSD")
            plt.legend(loc="best")
            plt.xlabel("Normalized spatial freq (0=DC, 1=Nyquist)")
            plt.ylabel("Radial PSD (normalized)")
            plt.yscale("log")
            image = wandb.Image(fig, caption=f"{phase} PSD: {self.wandb_name}")
            wandb.log({f"{phase}_psd": image})
            plt.close()
            self.plotted = True
        except Exception as e:
            print(f"Failed to plot {phase} psd: {e}")
            
    def plot_pred_hists(self, x, x_dirty, x_recon, x_dc, phase: str = "train"):
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(24., 6.))
            x = x.cpu().numpy()[0].squeeze().flatten()
            x_dirty = x_dirty.cpu().numpy()[0].squeeze().flatten()
            x_recon = x_recon.cpu().numpy()[0].squeeze().flatten()
            x_dc = x_dc.cpu().numpy()[0].squeeze().flatten()
            ax1.hist(x, color="k", density=True)
            ax2.hist(x_dirty, color="k", density=True)
            ax3.hist(x_recon, color="k", density=True)
            ax4.hist(x_dc, color="k", density=True)
            ax1.axvline(np.mean(x), color="r", ls="--")
            ax2.axvline(np.mean(x_dirty), color="r", ls="--")
            ax3.axvline(np.mean(x_recon), color="r", ls="--")
            ax4.axvline(np.mean(x_dc), color="r", ls="--")
            ax1.set_title("Clean")
            ax2.set_title("Dirty")
            ax3.set_title("Recreated")
            ax4.set_title("DC")
            image = wandb.Image(fig, caption=f"{phase} Histograms: {self.wandb_name}")
            wandb.log({f"{phase}_histograms": image})
            plt.close()
            self.plotted = True
        except Exception as e:
            print(f"Failed to plot {phase} histograms: {e}")
            
    def on_train_start(self):
        _ = self.denoise_loss._weight_schedule(self.current_epoch, None)

    def on_train_epoch_end(self) -> None:
        """
        Called by Lightning at the end of each training epoch.
        Here choose a random validation-batch index to plot next time during validation.
        """
        # 1) How many val batches do are there?
        num_val_batches = self.trainer.num_val_batches[0]
        
        self.log("epoch", self.current_epoch)

        # 2) Randomly choose one index in [0, num_val_batches)
        #    (If num_val_batches is 0, skip)
        if num_val_batches > 0:
            self.plot_on_batch = random.randrange(num_val_batches)
        else:
            self.plot_on_batch = None
            
    def on_train_epoch_start(self) -> None:
        """
        Called by Lightning at the start of each epoch.
        """
        if self.transform:
            self._make_aug()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def on_validation_epoch_start(self) -> None:
        """
        Called by Lightning right before validation begins.
        Reset the “plotted” flag so that validation_step can plot once.
        """
        self.plotted = False
        
        if self.scale_out > 0.:
            
            if self.current_epoch % 10 == 0 and self.current_epoch > 0 and self.grow_scale not in [0., 1.]:
                # grow scale by 10% every 10 epochs
                self.out_scale.data.mul_(self.grow_scale)
            
            self.log("out_scale", self.out_scale, prog_bar=False)
            
    def on_validation_epoch_end(self):
        
        if self.final_activation == "asinh":
            self.log("asinh_k", self.final_act.k)
        elif self.final_activation == "softplus":
            self.log("softplus_k", self.final_act.a)
            
        if self.current_epoch > self.ramp_epochs or not self.schedule_loss:
            scheduler = self.lr_schedulers()[1]
            try:
                total_val = self.trainer.callback_metrics.get("val/loss")
                if total_val is not None:
                    scheduler.step(total_val)
            except Exception:
                _ = None
            
        if self.schedule_loss:
            # Get loss stats
            loss_stats = {
                k.split("/")[-1]: float(v)
                for k, v in self.trainer.callback_metrics.items()
            }
            
            # Update weights
            new_weights = self.denoise_loss._weight_schedule(self.current_epoch, loss_stats)
            self.denoise_loss.weight_sum = sum(new_weights.values())
            
            # Log them for monitoring
            self.log_dict({f"weight/{k}": float(v) for k, v in new_weights.items()},
                          prog_bar=False, logger=True, on_epoch=True)
            
    def hann2d(self, H, W):
        wy = torch.hann_window(H, periodic=False, device=self.device)
        wx = torch.hann_window(W, periodic=False, device=self.device)
        return wy[:, None] * wx[None, :]

    def rfft_mag2(self, img, use_ortho=True, apodize=True):
        """
        img: (B,1,H,W) or (B,H,W); real
        returns Power = |FFT|^2 with DC removed, shape (B,H,W//2+1)
        """
        if img.ndim == 3: img = img.unsqueeze(1)
        B, _, H, W = img.shape
        x = img - img.mean(dim=(-2,-1), keepdim=True)          # remove DC bias
        if apodize:
            w = self.hann2d(H, W)
            x = x * w
        Xk = torch.fft.rfft2(x.squeeze(1), norm=("ortho" if use_ortho else "backward"))
        return (Xk.abs()**2)

    def radial_bins(self, H, W):
        ky = torch.fft.fftfreq(H, d=1.0, device=self.device)
        kx = torch.fft.rfftfreq(W, d=1.0, device=self.device)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        r = torch.sqrt(KY**2 + KX**2) / 0.5   # 0..1 (Nyquist=1)
        nb = min(H//2, W//2)
        edges = torch.linspace(0, 1, nb+1, device=self.device)
        centers = 0.5*(edges[1:]+edges[:-1])
        return r, edges, centers

    def radial_average(self, P, r, edges):
        """
        P: (B,H,W//2+1) power; r, edges from radial_bins
        returns psd: (B, nbins)
        """
        B = P.shape[0]
        nb = edges.numel()-1
        psd = torch.zeros(B, nb, device=self.device)
        for i in range(nb):
            m = (r >= edges[i]) & (r < edges[i+1])
            w = m.float()
            denom = w.sum().clamp_min(1.0)
            psd[:, i] = (P * w).sum(dim=(-2,-1)) / denom
        return psd

    def radial_psd(self, img, use_ortho=True, apodize=True, normalize=True):
        """
        returns f (0..1 of Nyquist), PSD (B, nbins) normalized per-image to max=1 if normalize=True
        """
        B, H, W = (img.shape[0], img.shape[-2], img.shape[-1]) if img.ndim==4 else (1, img.shape[-2], img.shape[-1])
        P = self.rfft_mag2(img, use_ortho=use_ortho, apodize=apodize)
        r, edges, centers = self.radial_bins(H, W)
        psd = self.radial_average(P, r, edges)
        if normalize:
            psd = psd / (psd.amax(dim=1, keepdim=True).clamp_min(1e-12))
        return centers.detach().cpu().numpy(), psd.detach().cpu().numpy()  # (nbins,), (B,nbins)

    # Beam roll-off from restoring PSF (centered, peak=1)
    def radial_otf_mag(self, psf_centered_peak1):
        """
        psf: (B,1,H,W) or (B,H,W); centered, peak=1
        returns f (0..1 Nyquist), |OTF| radial avg normalized to max=1.
        """
        psf = psf_centered_peak1
        if psf.ndim==3: psf = psf.unsqueeze(1)
        B,_,H,W = psf.shape
        psf0 = torch.fft.ifftshift(psf, dim=(-2,-1))
        Hk = torch.fft.rfft2(psf0.squeeze(1), norm="ortho").abs()
        r, edges, centers = self.radial_bins(H, W)
        roll = self.radial_average(Hk, r, edges)
        roll = roll / (roll.amax(dim=1, keepdim=True).clamp_min(1e-12))
        return centers.detach().cpu().numpy(), roll.detach().cpu().numpy()
                