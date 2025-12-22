from astropy.io import fits
import glob
import kornia as K
import math
import numpy as np
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def parse_name(filename: str):
    match = re.match(r"([a-zA-Z]+)(\d+)_", filename)
    if match:
        prefix, run_id = match.groups()
        return prefix, int(run_id)
    return None, None

def normalize_image(img, normalize: str = "tanh_range",
                    softsign_a: float = 3.,
                    atan_s: float = 3.,
                    clamp_k: float = 6.,
                    asinh_s: float = 1.5,
                    asinh_k: float = 8.,
                    ):
    
    # normalization
    if normalize in {"norm", "tanh", "tanh_range"}:
        img -= img.min()
        denom = img.max() + 1e-12 * abs(img.max())
        img /= denom
        if normalize == "tanh_range":
            img = img * 2.0 - 1.0
        elif normalize == "tanh":
            img = np.tanh((img - 1.))  # gentle squash; 3.0 helps use [-1,1] without saturating everything
    elif normalize == "softsign":
        img = (img / softsign_a) / (1 + np.abs(img / softsign_a))
    elif normalize == "atan":
        img = (2/np.pi) * np.arctan(img / atan_s) 
    elif normalize == "clamp":
        img = np.clip(img, -clamp_k, clamp_k) / clamp_k 
    elif normalize == "asinh":
        img = np.clip(np.arcsinh(img / asinh_s) / np.arcsinh(asinh_k / asinh_s), -1, 1)
        
    return img

def corner_rms(x, frac=0.15):
    H, W = x.shape
    m = max(1, int(min(H, W)*frac))
    tiles = np.concatenate([x[:m,:m].ravel(), x[:m,-m:].ravel(),
                            x[-m:,:m].ravel(), x[-m:,-m:].ravel()])
    med = np.median(tiles)
    rms = 1.4826*np.median(np.abs(tiles - med))
    rms = rms if rms != 0 else 1
    return rms

def blur_with_beam(img, sig_maj, sig_min, phi_img):
    # img: [B,1,H,W]
    deg = math.degrees(phi_img)
    x = K.geometry.transform.rotate(img, -deg, align_corners=False)  # align major axis with x
    kx = max(3, 2*int(3*sig_maj)+1)
    ky = max(3, 2*int(3*sig_min)+1)
    x = F.gaussian_blur(x, kernel_size=[ky, kx], sigma=[max(sig_min,1e-6), max(sig_maj,1e-6)])
    x = K.geometry.transform.rotate(x, +deg, align_corners=False)
    return x

def recenter_peak1(psf: np.ndarray, contract: str = "center", eps=1e-12):
    """
    contract: "center" (peak at [H//2,W//2]) or "ifft" (peak at [0,0]).
    """
    psf = torch.tensor(psf)
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
    return x.numpy().squeeze() if psf.dim()==4 else x.squeeze(1).numpy()
    
def make_psf_from_header(H: int, W: int, hdr: dict, norm: str = "peak"):
    """
    Build a *centered* elliptical Gaussian PSF from FITS header keywords.
    - BMAJ, BMIN in *degrees* (FWHM along major/minor)
    - BPA  in *degrees East of North*
    - Pixel scale from either:
        * CDELT2 (deg/pix)  → arcsec/pix = |CDELT2| * 3600
        * AU_PP & DIST (pc) → arcsec/pix = AU_PP / DIST
    Returns
    -------
    psf_centered : (H, W) float32, peak at image center (H//2, W//2)
    meta         : dict with FWHM (px/mas), px_per_beam, cell_mas, etc.
    """
    # --- header values
    bmaj_deg = float(hdr.get("BMAJ", 0.0))
    bmin_deg = float(hdr.get("BMIN", 0.0))
    bpa_deg  = float(hdr.get("BPA" , 0.0))
    dist_pc  = float(hdr.get("DIST", 140.0))
    au_pp    = float(hdr.get("AU_PP", 1.0))

    if bmaj_deg <= 0 or bmin_deg <= 0:
        raise ValueError("BMAJ/BMIN must be > 0 (degrees).")

    # pixel scale (arcsec/pix)
    if "CDELT2" in hdr and "AU_PP" not in hdr:
        arcsec_per_pix = abs(float(hdr["CDELT2"])) * 3600.0
    else:
        arcsec_per_pix = au_pp / max(dist_pc, 1e-12)

    cell_mas = 1000.0 * arcsec_per_pix

    # FWHM arcsec → pixels; then σ = FWHM / (2√(2ln2))
    fwhm_maj_pix = (bmaj_deg * 3600.0) / max(arcsec_per_pix, 1e-30)
    fwhm_min_pix = (bmin_deg * 3600.0) / max(arcsec_per_pix, 1e-30)
    sig_maj_pix  = fwhm_maj_pix / 2.354820045
    sig_min_pix  = fwhm_min_pix / 2.354820045

    # grid & rotation (BPA: East of North ⇒ θ = BPA)
    yy, xx = np.meshgrid(np.arange(H) - H//2, np.arange(W) - W//2, indexing="ij")
    th = np.deg2rad(bpa_deg)
    xr =  xx*np.cos(th) - yy*np.sin(th)
    yr =  xx*np.sin(th) + yy*np.cos(th)

    # elliptical Gaussian: major along 'yr'
    g = np.exp(-0.5 * ((xr / (sig_min_pix + 1e-30))**2 +
                       (yr / (sig_maj_pix + 1e-30))**2)).astype(np.float64)

    px_per_beam = float(2.0 * np.pi * sig_maj_pix * sig_min_pix)

    # normalization
    if norm == "peak":
        cy, cx = H//2, W//2
        c = abs(g[cy, cx]);  c = c if c > 0 else np.max(np.abs(g))
        g = g / (c if c > 0 else 1.0)
    elif norm == "sum":
        s = g.sum(dtype=np.float64)
        g = g / (s if s > 0 else 1.0)
    else:
        raise ValueError("norm must be 'peak' or 'sum'.")

    psf_centered = g.astype(np.float32)

    meta = dict(
        FWHM_MAJOR_PX=float(fwhm_maj_pix),
        FWHM_MINOR_PX=float(fwhm_min_pix),
        FWHM_MAJOR_MAS=float(fwhm_maj_pix * cell_mas),
        FWHM_MINOR_MAS=float(fwhm_min_pix * cell_mas),
        PX_PER_BEAM=px_per_beam,
        CELL_MAS=cell_mas,
        BMAJ_DEG=bmaj_deg, BMIN_DEG=bmin_deg, BPA_DEG=bpa_deg,
        NORM=norm,
    )
    return psf_centered, meta

def make_otf_from_psf(
    psf,
    image_size: int,
    device=None,
    dtype=torch.float32,
    use_ortho: bool = True,
    psf_is_centered: bool = True,   # <- set True for your pipeline
    reassert_peak: bool = True,     # <- keep center exactly +1 after pad/crop
    ):
    """
    psf: (H,W) or (B,H,W) or (B,1,H,W)
         EXPECTED to be *centered* (peak in the middle) if psf_is_centered=True.
         If you pass an ifft-shifted PSF, set psf_is_centered=False.
    Returns: (B,1,H,W//2+1) complex OTF
    """
    psf = torch.as_tensor(psf, dtype=dtype, device=device)
    if psf.ndim == 2:
        psf = psf.unsqueeze(0).unsqueeze(0)         # (1,1,h,w)
    elif psf.ndim == 3:
        psf = psf.unsqueeze(1)                      # (B,1,h,w)
    elif psf.ndim != 4:
        raise ValueError("psf must be (H,W), (B,H,W), or (B,1,H,W)")

    B, _, h, w = psf.shape
    H = W = int(image_size)

    # If the PSF came in ifft-shifted, convert it to centered form for padding.
    if not psf_is_centered:
        psf = torch.fft.fftshift(psf, dim=(-2, -1))

    # Recenter to global |peak| to be robust to small drifts.
    # This makes sure the absolute maximum sits at the middle BEFORE normalization.
    with torch.no_grad():
        flat = psf.abs().reshape(B, -1)
        idx = flat.argmax(dim=1)
        iy, ix = (idx // w).tolist(), (idx % w).tolist()
        for b in range(B):
            dy = iy[b] - h // 2
            dx = ix[b] - w // 2
            if dy or dx:
                psf[b, 0] = torch.roll(psf[b, 0], shifts=(-dy, -dx), dims=(-2, -1))

    # Re-assert peak=+1 at the center (recommended for peak-1 convention)
    if reassert_peak:
        center = psf[..., h // 2, w // 2]
        sign = torch.sign(center); sign[sign == 0] = 1.0
        psf = psf * sign.unsqueeze(-1).unsqueeze(-1)
        den = psf[..., h // 2, w // 2].abs().clamp_min(1e-12)
        psf = psf / den.unsqueeze(-1).unsqueeze(-1)

    # Center-crop if needed, then symmetric pad into the big array
    if H < h or W < w:
        top = (h - H) // 2; left = (w - W) // 2
        psf = psf[..., top:top + H, left:left + W]
        h, w = H, W

    out = torch.zeros((B, 1, H, W), dtype=dtype, device=device)
    y0 = H // 2 - h // 2
    x0 = W // 2 - w // 2
    out[:, :, y0:y0 + h, x0:x0 + w] = psf

    # Shift once so the impulse is at [0,0] before the FFT
    out = torch.fft.ifftshift(out, dim=(-2, -1))

    norm = "ortho" if use_ortho else "backward"
    otf_r = torch.fft.rfft2(out, norm=norm)  # (B,1,H,W//2+1) complex
    return otf_r

class PairedFitsDataset(Dataset):
    def __init__(self, 
                 clean_fits, 
                 dirty_fits,
                 psf,
                 otf,
                 keep_original: bool = False,
                 dtype: torch.dtype = torch.float32,
                 names: np.ndarray = None,
                 ):
        self.x = torch.tensor(clean_fits, dtype=dtype)
        self.y = torch.tensor(dirty_fits, dtype=dtype)
        self.psf = torch.tensor(psf, dtype=dtype)  
        self.otf = torch.tensor(otf)
        self.names = names

        self.dtype = dtype
        
        if keep_original:
            self.clean, self.dirty = clean_fits, dirty_fits
            
    def write_files(self, file_path):
        np.savetxt(file_path, self.names, delimiter=",", fmt="%s")
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return {"clean": self.x[idx],
               "dirty": self.y[idx],
               "psf": self.psf[idx],
               "otf": self.otf[idx]
               }
    
def load_fits_with_psf(
    fits_file,
    normalize: str = "tanh",
    add_channel: bool = True,
    clip: bool = True,
    clip_percentile: float = 100.0,
    return_otf: bool = True,
    noise_norm: bool = True,
    ):
    """
    Returns:
        dict with keys:
          image: np.float32 array (C,H,W) if add_channel else (H,W)
          psf:   np.float32 array (C,H,W) or (H,W), unit-sum
          otf:   np.complex64 array (H, W//2+1) if return_otf=True
          header: primary header (astropy Header)
          meta:  dict with useful beam/pixel metadata
    """
    hdul = fits.open(fits_file)
    # --- image ---
    img = hdul[0].data
    img = np.asarray(img).squeeze()
    while img.ndim > 2:
        img = img[0, ...]
    img = img.astype(np.float32)
    H, W = img.shape
    hdr = hdul[0].header

    # optional clipping
    if clip and clip_percentile < 100.0:
        lo, hi = np.percentile(img, [100.0 - clip_percentile, clip_percentile])
        img = np.clip(img, lo, hi, out=img)
        
    if noise_norm:
        img = img / corner_rms(img)

    img = normalize_image(img, normalize)
        
    # add channel dim
    if add_channel:
        img = np.expand_dims(img, axis=0)  # (1,H,W)

    # --- PSF: prefer PSF extension if present; else rebuild from header ---
    psf = None
    for hdu in hdul[1:]:
        if isinstance(hdu, fits.ImageHDU) and (hdu.header.get("EXTNAME", "").upper() == "PSF"):
            psf = np.asarray(hdu.data, dtype=np.float32)
            break

    meta = {}
    hdr_vals = dict((k, v) for k, v in hdr.items())
    if psf is None:
        psf, meta = make_psf_from_header(H, W, hdr)
    else:
        # ensure unit sum + float32
        psf = psf.astype(np.float32)
        # s = psf.sum(dtype=np.float64)
        # if s > 0:
            # psf /= s
    if np.max(psf) > 1.1 or np.min(psf) < -1.1:
        breakpoint()
    # psf = recenter_and_peak1(psf)

    # optional OTF (for DC / forward loss)
    otf = None
    if return_otf:
        # compute OTF for (H,W) PSF; caller can broadcast as needed
        otf = make_otf_from_psf(psf, H).squeeze()
        if add_channel:
            otf = np.expand_dims(otf, axis=0)

    if add_channel:
        psf = np.expand_dims(psf, axis=0)  # (1,H,W)

    return {
        "image": img,     # dirty or clean, depending on file
        "psf": psf,
        "otf": otf,
        "header": hdr_vals,
        "meta": meta,
    }
    
def load_fits(fits_file, normalize: str = "tanh_range", 
              add_channel: bool = True, 
              clip: bool = True, clip_percentile: float = 99.5):
    
    with fits.open(fits_file) as hdul:
        fits_data = hdul[0].data
        fits_data = fits_data.squeeze()
        while len(fits_data.shape) > 2:
            fits_data = fits_data[0, ...]
        if clip:
            low = np.percentile(fits_data, 100. - clip_percentile)
            high = np.percentile(fits_data, clip_percentile)
            fits_data = np.clip(fits_data, low, high)
        fits_data = normalize_image(fits_data, normalize)
        if add_channel:
            fits_data = np.expand_dims(fits_data, axis=0)
    return fits_data
    
def load_fits_dir(data_path: str, normalize: str = "tanh_range", 
                  add_channel: bool = True,
                  clip: bool = True, clip_percentile: float = 99.5,
                  add_planet: bool = False, planet_path: str = "./data/data_cube.dat",
                  add_psf: bool = True,
                  ):
    fits_files = glob.glob(data_path + '/*.fits')
    fits_files.sort()
    data = []
    planet = []
    psf = []
    dist = []
    if add_planet or add_psf:
        df = pd.read_csv(planet_path, 
                    sep="\t", 
                )
    for fits_file in fits_files:
        if add_planet:
            _, run_name = parse_name(fits_file)
            this_df = df[df["Run"] == run_name]
        if not add_psf:
            data.append(load_fits(fits_file, normalize=normalize, 
                                add_channel=add_channel, 
                                clip=clip, clip_percentile=clip_percentile))
        else:
            fits_info = load_fits_with_psf(fits_file, normalize=normalize, 
                                add_channel=add_channel, 
                                clip=clip, clip_percentile=clip_percentile)
            data.append(fits_info["image"])
            psf.append(fits_info["psf"])
            dist.append(fits_info["header"].get("DIST", 140))
        if add_planet and "N_planet" in this_df.columns:
            n_planets = this_df["N_planet"].values[0]
            planet.append(n_planets)
        elif add_planet and "N_planet" not in this_df.columns:
            planet.append(0)
            
    return_info = {}
    data = np.array(data)
    return_info["data"] = data
    return_info["files"] = fits_files
    if add_psf:
        psf = np.array(psf)
        dist = np.array(dist)
        return_info["psf"] = psf
        return_info["dist"] = dist
    if add_planet:
        planet = np.array(planet)
        return_info["planet"] = planet
    return return_info if (add_psf or add_planet) else data

def load_paired_fits(data_path: str, 
                     normalize: str = "tanh_range",
                    add_channel: bool = True,
                    clip: bool = True,
                    return_paths: bool = False,
                    clean_clip_percentile: float = 100.0,
                    dirty_clip_percentile: float = 100.0,
                    dirty_folder: str = "dirty",
                    clean_folder: str = "clean",
                    return_otf: bool = True,
                    debug: bool = False,
                    add_planet: bool = False,
                    planet_path: str = "./data/data_cube.dat",
                    dirty_ext: str = "_dirty.fits",
                    clean_ext: str = "_clean.fits"
                    ):
    # 1. List all .fits in each subfolder
    clean_paths = glob.glob(os.path.join(data_path, clean_folder, "*.fits"))
    dirty_ext = "_dirty.fits"
    dirty_paths = glob.glob(os.path.join(data_path, dirty_folder, "*.fits"))

    # 2. Extract base names (strip off "_clean.fits" or "_dirty.fits")
    def base_from_clean(full_path: str) -> str:
        fname = os.path.basename(full_path)
        if not fname.endswith(clean_ext):
            return None
        return fname[: -len(clean_ext)]

    def base_from_dirty(full_path: str) -> str:
        fname = os.path.basename(full_path)
        if not fname.endswith(dirty_ext):
            return None
        return fname[: -len(dirty_ext)]

    clean_bases = {base_from_clean(p) for p in clean_paths if base_from_clean(p) is not None}
    dirty_bases = {base_from_dirty(p) for p in dirty_paths if base_from_dirty(p) is not None}

    # 3. Intersection of bases
    common_bases = clean_bases.intersection(dirty_bases)

    # 4. Re-build only those filenames that exist in both
    clean_files = [
        os.path.join(data_path, clean_folder, base + clean_ext)
        for base in sorted(common_bases)
    ]
    dirty_files = [
        os.path.join(data_path, dirty_folder, base + dirty_ext)
        for base in sorted(common_bases)
    ]
    
    if debug:
        clean_files = clean_files[:100]
        dirty_files = dirty_files[:100]

    # 5. Load each FITS file via your load_fits function
    clean_imgs = [
        load_fits_with_psf(f, normalize=normalize, add_channel=add_channel, clip=clip, 
                           clip_percentile=clean_clip_percentile, return_otf=return_otf,)
        for f in clean_files
    ]
    dirty_imgs = [
        load_fits_with_psf(f, normalize=normalize, add_channel=add_channel, 
                           clip=clip, 
                           clip_percentile=dirty_clip_percentile, return_otf=return_otf,)
        for f in dirty_files
    ]
    
    if add_planet:
        df = pd.read_csv(planet_path, 
                sep="\t", 
            )
        planet = []
        for f in clean_files:
            try:
                _, run_name = parse_name(f)
                this_df = df[df["Run"] == run_name]
                planet.append(this_df["N_planet"].values[0])
            except:
                planet.append(0)
    else:
        planet = None
            
    return {"clean_imgs": clean_imgs,
            "dirty_imgs": dirty_imgs,
            "clean_files": clean_files if return_paths else None,
            "dirty_files": dirty_files if return_paths else None,
            "planet": planet if add_planet else None,
            }
  
def make_all_paired_datasets(data_path: str, 
                        train_split: float = 0.8,
                        val_split: float = 0.2,
                        normalize: str = "tanh",
                        keep_original: bool = False,
                        add_channel: bool = True,
                        clip: bool = True, 
                        clean_clip_percentile: float = 100.0,
                        dirty_clip_percentile: float = 100.,
                        dirty_folder: str = "dirty",
                        clean_folder: str = "clean",
                        dtype: torch.dtype = torch.float32,
                        return_otf: bool = True,
                        planet_path: str = "./data/data_cube.dat",
                        ):
    
    return_data = load_paired_fits(data_path, normalize=normalize, 
                                   add_channel=add_channel, clip=clip, 
                                   clean_clip_percentile=clean_clip_percentile, dirty_clip_percentile=dirty_clip_percentile,
                                   return_paths=True,
                                   dirty_folder=dirty_folder, clean_folder=clean_folder, planet_path=planet_path,
                                   return_otf=return_otf,)
    clean_fits = np.array([f["image"] for f in return_data["clean_imgs"]])
    psf = np.array([f["psf"] for f in return_data["dirty_imgs"]])
    otf = np.array([f["otf"] for f in return_data["dirty_imgs"]])
    dirty_fits = np.array([f["image"] for f in return_data["dirty_imgs"]])
    dirty_files = np.array(return_data["dirty_files"])

    X_train, X_test,\
    y_train, y_test,\
    psf_train, psf_test,\
    otf_train, otf_test,\
    file_train, file_test = train_test_split(clean_fits, dirty_fits,
                                             psf, otf, 
                                             dirty_files, 
                                             test_size=1.-train_split, random_state=123,)
    X_train, X_val,\
    y_train, y_val,\
    psf_train, psf_val,\
    otf_train, otf_val,\
    file_train, file_val = train_test_split(X_train, y_train, 
                                            psf_train, otf_train, 
                                            file_train, 
                                            test_size=val_split, random_state=123,)
    return {"train": PairedFitsDataset(X_train, y_train, psf_train, otf_train, 
                                       dtype=dtype,
                                       names=file_train,
                                       ),
            "val": PairedFitsDataset(X_val, y_val, psf_val, otf_val, 
                                     dtype=dtype,
                                     names=file_val
                                     ),
            "test": PairedFitsDataset(X_test, y_test, psf_test, otf_test, 
                                      keep_original=keep_original,
                                      dtype=dtype,
                                      names=file_test,
                                      ),
            }
