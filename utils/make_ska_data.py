import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
import os
import yaml

# ---------------------------
# Small helpers (unit-safe)
# ---------------------------
SKA_MID_LAT_DEG = -30.712925  # from your COFA
JY_TO_W = 1e-26
mJy_TO_W = 1e-29
UJy_TO_W = 1e-32  # µJy → W m^-2 Hz^-1

### from Ilee et al. (2020)
SKA_PRESETS = {
    # resolution, hours : (rms_uJy, peak_snr, vis_noise_mJy, radial_exp, uv_points, gamma)
    ("34mas", 10):  dict(rms=0.83, peak_snr=4.90,  vis=0.56, radial_exp=1.6, uv_points=200_000, gamma=0.70),
    ("34mas", 100):  dict(rms=0.26, peak_snr=7.73,  vis=0.56, radial_exp=1.6, uv_points=200_000, gamma=0.70),
    ("34mas", 1000): dict(rms=0.08, peak_snr=16.80, vis=0.56, radial_exp=1.5, uv_points=800_000, gamma=0.75),
    ("67mas", 10):  dict(rms=0.44, peak_snr=7.58, vis=0.28, radial_exp=1.2, uv_points=160_000, gamma=0.90),
    ("67mas", 100):  dict(rms=0.14, peak_snr=17.17, vis=0.28, radial_exp=1.2, uv_points=160_000, gamma=0.90),
    ("67mas", 1000): dict(rms=0.05, peak_snr=44.75, vis=0.28, radial_exp=1.25, uv_points=600_000, gamma=0.90),
}

def ujy_to_W(rms_uJy):  # µJy/beam → W m^-2 Hz^-1 / beam
    return float(rms_uJy) * 1e-32

def mas_to_rad(mas):    # mas → radians
    return np.deg2rad(float(mas) / 3_600_000.0)

def sanitize_finite(a):
    b = np.array(a, dtype=np.float64, copy=True)
    b[~np.isfinite(b)] = 0.0
    return b

def au_per_pix_to_mas(au_per_pix, distance_pc):
    """
    Convert physical resolution (au/pixel) to angular resolution (mas/pixel).
    """
    return (au_per_pix / distance_pc) * 1000.0  # mas

def robust_std(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def corner_rms(x, frac=0.15):
    H, W = x.shape
    m = max(1, int(min(H, W)*frac))
    tiles = np.concatenate([x[:m,:m].ravel(), x[:m,-m:].ravel(),
                            x[-m:,:m].ravel(), x[-m:,-m:].ravel()])
    med = np.median(tiles)
    return 1.4826*np.median(np.abs(tiles - med))

def needed_baseline_scale(ant_ENU_m, freq_GHz, target_fwhm_mas):
    c = 299_792_458.0
    lam = c / (freq_GHz*1e9)
    # current Bmax
    diffs = ant_ENU_m[:,None,:] - ant_ENU_m[None,:,:]
    B = np.sqrt((diffs[...,0]**2 + diffs[...,1]**2 + diffs[...,2]**2)).max()
    # target baseline for desired FWHM
    theta = np.deg2rad(target_fwhm_mas/3_600_000.0)   # mas → rad
    B_needed = lam / theta
    return float(B_needed / max(B, 1e-9)), float(B/1e3), float(B_needed/1e3)  # (scale, Bcur[km], Bneed[km])

def check_grid_support(cell_mas, img_W, freq_GHz):
    c = 299_792_458.0
    lam = c / (freq_GHz*1e9)
    dtheta = np.deg2rad(cell_mas/3_600_000.0)
    B_grid = lam / (2*dtheta)          # meters
    return B_grid/1e3                   # km

def declination_bounds(lat_deg=SKA_MID_LAT_DEG, h_min_deg=20.0):
    """Bounds in declination ensuring the source transits above h_min."""
    span = 90.0 - float(h_min_deg)
    dmin = max(-90.0, float(lat_deg) - span)
    dmax = min(+90.0,  float(lat_deg) + span)
    return dmin, dmax

def sample_declinations(n, lat_deg=SKA_MID_LAT_DEG, h_min_deg=20.0, rng=None):
    """Sample n declinations uniformly in sky area over the visible band."""
    dmin, dmax = declination_bounds(lat_deg, h_min_deg)
    r = np.random.default_rng(rng)
    u = r.uniform(np.sin(np.radians(dmin)), np.sin(np.radians(dmax)), size=n)
    return np.degrees(np.arcsin(u))  # degrees

# ---------------------------
# Hour angles (elevation-limited track)
# ---------------------------
def hour_angles_for_track(dec_deg, lat_deg, t_hr, dt_s=30.0, el_min_deg=20.0):
    dec = np.deg2rad(dec_deg); lat = np.deg2rad(lat_deg)
    s_el = np.sin(np.deg2rad(el_min_deg))
    x = np.clip((s_el - np.sin(lat)*np.sin(dec)) / (np.cos(lat)*np.cos(dec) + 1e-30), -1, 1)
    Hmax_el  = np.arccos(x)
    Hmax_req = np.pi * (t_hr / 24.0)
    Hmax = min(Hmax_el, Hmax_req)
    nstep = max(2, int(np.ceil((2*Hmax)/(2*np.pi) * (24*3600/dt_s))))
    return np.linspace(-Hmax, +Hmax, nstep, dtype=np.float64)


# ---------------------------
# UV coverage from array [E,N,U] (meters)
# ---------------------------
def array_to_uv_counts(
    ant_ENU_m, dec_deg, freq_GHz, H_grid_rad,
    img_shape, cell_mas, blur_sigma_px=1.25
):
    H, W = img_shape
    lam_m = 299_792_458.0 / (freq_GHz * 1e9)
    dec = np.deg2rad(dec_deg)

    ant = np.asarray(ant_ENU_m, dtype=np.float64)
    E = ant[:,0][:,None] - ant[:,0][None,:]
    N = ant[:,1][:,None] - ant[:,1][None,:]
    U = ant[:,2][:,None] - ant[:,2][None,:]
    iu, ju = np.triu_indices(len(ant), k=1)
    bE = E[iu,ju]; bN = N[iu,ju]; bU = U[iu,ju]

    sH = np.sin(H_grid_rad)[None,:]; cH = np.cos(H_grid_rad)[None,:]
    sD = np.sin(dec);               cD = np.cos(dec)

    u = ( bE[:,None]*sH + bN[:,None]*cH ) / lam_m
    v = ( -bE[:,None]*sD*cH + bN[:,None]*sD*sH + bU[:,None]*cD ) / lam_m

    dtheta = mas_to_rad(cell_mas)             # rad/pixel
    du = 1.0 / (W * dtheta)                   # wavelengths/pixel
    dv = 1.0 / (H * dtheta)

    uc, vc = W//2, H//2
    ui = np.rint(u/du).astype(int) + uc
    vi = np.rint(v/dv).astype(int) + vc

    counts = np.zeros((H, W), dtype=np.float64)
    m = (ui>=0)&(ui<W)&(vi>=0)&(vi<H)
    ui = ui[m]; vi = vi[m]
    np.add.at(counts, (vi, ui), 1.0)
    # conjugate
    ui2 = 2*uc - ui; vi2 = 2*vc - vi
    m2 = (ui2>=0)&(ui2<W)&(vi2>=0)&(vi2<H)
    np.add.at(counts, (vi2[m2], ui2[m2]), 1.0)

    # small uv gridding blur (separable; stable)
    counts = gaussian_blur_uv_safe(counts, sigma_px=blur_sigma_px)
    # enforce symmetry
    counts = 0.5*(counts + np.flipud(np.fliplr(counts)))
    # kill DC explicitly (no zero-spacing)
    counts[vc, uc] = 0.0

    return counts

def gaussian_blur_uv_safe(mask_counts, sigma_px=1.25):
    arr = sanitize_finite(mask_counts)
    rad = int(max(1, np.ceil(3.0 * float(sigma_px))))
    x = np.arange(-rad, rad+1, dtype=np.float64)
    g = np.exp(-0.5 * (x/float(sigma_px))**2)
    if g.sum() != 0: g /= g.sum()
    # rows
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i,:] = np.convolve(arr[i,:], g, mode="same")
    # cols
    tmp = np.empty_like(out)
    for j in range(out.shape[1]):
        tmp[:,j] = np.convolve(out[:,j], g, mode="same")
    return tmp


def scale_noise_to_map_rms(noise_raw, rms_uJy_target):
    """
    Scale a realized residual field (unitless) so that its *measured* RMS
    equals the requested map RMS in W m^-2 Hz^-1 / beam.
    """
    # 1) convert target to W units
    sigma_W = float(rms_uJy_target) * UJy_TO_W

    # 2) measure current RMS of the raw field (unitless)
    raw = corner_rms(noise_raw)
    if raw == 0 or not np.isfinite(raw):
        raise RuntimeError(f"noise_raw RMS invalid: {raw}")

    # 3) scale
    s = sigma_W / raw
    noise = noise_raw * s

    # 4) self-checks (print once while debugging)
    meas_W   = corner_rms(noise)
    meas_uJy = meas_W * 1e32
    # print(f"[scale_noise] target={rms_uJy_target:.6g} µJy  "
    #       f"rawRMS={raw:.3e} (unitless)  scale={s:.3e}  "
    #       f"meas={meas_uJy:.6g} µJy")

    # sanity: within 2%
    if not (np.isfinite(meas_W) and abs(meas_W - sigma_W) <= 0.02*sigma_W):
        raise AssertionError("Noise RMS scaling failed (unit mismatch or stray conversion).")
    return noise

def force_unit_rms(noise_raw):
    """
    Normalize an arbitrary residual field to unit RMS using a robust estimator.
    Returns noise_unit (RMS≈1) and the measured raw RMS.
    """
    raw = corner_rms(noise_raw)
    if not np.isfinite(raw) or raw == 0:
        raise RuntimeError(f"[force_unit_rms] invalid raw RMS: {raw}")
    return (noise_raw / raw).astype(np.float64), float(raw)

def scale_noise_and_model(noise_raw, model0, rms_uJy_target, peak_snr):
    """
    Make noise RMS = sigma_W and model peak = peak_snr * sigma_W exactly (up to fp error).
    Assumes model0 is the already-convolved model (dirty or restoring), arbitrary scale.
    """
    # 1) target map RMS in W/beam (CONVERT ONCE)
    sigma_W = float(rms_uJy_target) * UJy_TO_W

    # 2) unit-RMS residuals, then scale to sigma_W
    noise_unit, raw = force_unit_rms(noise_raw)
    noise = noise_unit * sigma_W

    # 3) set model peak from the *target* sigma (not re-measured)
    m0max = float(model0.max())
    if not np.isfinite(m0max) or m0max <= 0:
        raise RuntimeError(f"[scale_noise_and_model] invalid model0.max={m0max}")
    peak_W  = float(peak_snr) * sigma_W
    model   = (model0 * (peak_W / m0max)).astype(np.float64)

    # 4) self-check (prints once; comment out after you see it’s correct)
    meas_W   = corner_rms(noise)
    meas_uJy = meas_W * 1e32
    snr_meas = (model + noise).max() / (corner_rms(noise) or 1.0)
    print(f"[scale] target σ={rms_uJy_target:.6g} µJy → {sigma_W:.3e} W | "
          f"rawRMS={raw:.3e} | meas σ={meas_uJy:.6g} µJy | SNR target={peak_snr}, meas≈{snr_meas:.2f}")

    # sanity: within 2%
    if not (np.isfinite(meas_W) and abs(meas_W - sigma_W) <= 0.02 * sigma_W):
        raise AssertionError("[scale_noise_and_model] noise RMS scaling failed (unit/extra-scaling mismatch).")

    return noise.astype(np.float32), model.astype(np.float32), sigma_W, peak_W

def pre_noise_clip_model(model0_W, sigma_beam_W, method="asinh",
                         k=15.0, s=3.0, gamma=0.5):
    """
    model0_W: beam-convolved clean (W m^-2 Hz^-1 / beam)
    sigma_beam_W: target map RMS in W m^-2 Hz^-1 / beam
    method: "hard", "asinh", or "power"
      hard : clamp at ±k σ  (linear in-range, hard edges)
      asinh: smooth compression; linear for |S|≲s, gentle roll-off to ~log
      power: signed power-law |S|^γ (γ<1 compresses highs), linear at zero
    k:     saturation level in σ units (for hard/asinh)
    s:     bend scale in σ units (for asinh)
    gamma: power exponent in (0,1] (for power)
    """
    S = model0_W / (sigma_beam_W + 1e-30)   # SNR map

    if method == "hard":
        S2 = np.clip(S, -k, k)

    elif method == "asinh":
        # linear for small S, gentle compression for large S
        denom = np.arcsinh(k / s)
        S2 = np.arcsinh(S / s) / denom
        S2 *= k   # so that |S2| ≤ k and small S2≈S

    elif method == "power":
        # signed power-law compression; choose γ in [0.4, 0.8]
        S2 = np.sign(S) * (np.abs(S) ** gamma)

    else:
        raise ValueError("method must be 'hard', 'asinh', or 'power'.")

    return S2 * sigma_beam_W 

def psf_from_uv_mask(mask, out="center"):
    """
    mask: uv weights on the FFT grid (DC at center)
    out : "center" (peak at center) or "ifft" (peak at [0,0])
    """
    M = np.asarray(mask, np.float64)
    # enforce Hermitian so PSF is real
    M = 0.5*(M + np.flipud(np.fliplr(M)))

    # dirty beam in ifft convention (peak should be near [0,0])
    psf_ifft = np.fft.ifft2(np.fft.ifftshift(M), norm="ortho").real

    # recenter to the true |peak|
    iy, ix = np.unravel_index(np.argmax(np.abs(psf_ifft)), psf_ifft.shape)
    psf_ifft = np.roll(psf_ifft, shift=(-iy, -ix), axis=(0,1))

    # make the peak +1 at [0,0]
    c = psf_ifft[0, 0]
    psf_ifft *= (1.0 if c >= 0 else -1.0) / max(abs(c), 1e-12)

    if out == "ifft":
        return psf_ifft.astype(np.float32)             # peak at [0,0]
    elif out == "center":
        return np.fft.fftshift(psf_ifft).astype(np.float32)  # peak in middle
    else:
        raise ValueError("out must be 'center' or 'ifft'")

def measure_psf_moments(psf):
    H, W = psf.shape
    yy, xx = np.meshgrid(np.arange(H)-H//2, np.arange(W)-W//2, indexing="ij")
    w = psf - psf.min(); w[w<0] = 0
    s = w.sum()
    if s == 0:
        return dict(fwhm_major=0., fwhm_minor=0., bpa=0., px_per_beam=0.)
    mx = (w*xx).sum()/s; my = (w*yy).sum()/s
    dx, dy = xx-mx, yy-my
    vxx = (w*dx*dx).sum()/s; vyy = (w*dy*dy).sum()/s; vxy = (w*dx*dy).sum()/s
    t = vxx+vyy; d = vxx-vyy; r = np.hypot(0.5*d, vxy)
    lam_max = max(0.5*(t + 2*r), 0.); lam_min = max(0.5*(t - 2*r), 0.)
    smaj = np.sqrt(lam_max); smin = np.sqrt(lam_min)
    fmaj = 2.354820045 * smaj; fmin = 2.354820045 * smin
    theta_x = 0.5*np.arctan2(2*vxy, d)
    bpa_eon = (90.0 - np.degrees(theta_x)) % 180.0
    px_beam = float(2.0 * np.pi * smaj * smin)
    return dict(fwhm_major=fmaj, fwhm_minor=fmin, bpa=bpa_eon, px_per_beam=px_beam)

def gaussian_restoring_beam(H, W, fwhm_major_mas, fwhm_minor_mas, bpa_eon_deg, cell_mas):
    yy, xx = np.meshgrid(np.arange(H)-H//2, np.arange(W)-W//2, indexing="ij")
    th = np.deg2rad(90.0 - bpa_eon_deg)
    xr =  xx*np.cos(th) - yy*np.sin(th)
    yr =  xx*np.sin(th) + yy*np.cos(th)
    smaj = (fwhm_major_mas / cell_mas) / (2.0*np.sqrt(2.0*np.log(2.0)))
    smin = (fwhm_minor_mas / cell_mas) / (2.0*np.sqrt(2.0*np.log(2.0)))
    g = np.exp(-0.5*((xr/smin)**2 + (yr/smaj)**2)).astype(np.float64)
    g /= g.max() if g.max()!=0 else 1.0   # peak=1
    return g.astype(np.float32)


def measure_beam_core(psf_peak1, cell_mas, frac=0.5, max_radius_px=64):
    """
    psf_peak1: PSF normalized to peak=1 (dirty or restoring)
    Returns FWHM_major/minor in *pixels* and *mas*, plus BPA (deg, N→E).
    """
    psf = np.asarray(psf_peak1, dtype=np.float64)
    H, W = psf.shape
    vc, uc = H//2, W//2

    # central mask: above frac*peak and within radius to avoid sidelobes/pedestal
    yy, xx = np.meshgrid(np.arange(H)-vc, np.arange(W)-uc, indexing="ij")
    r2 = xx*xx + yy*yy
    core = (psf >= frac * psf[vc, uc]) & (r2 <= max_radius_px**2)

    # weights: just the PSF inside the core
    w = psf * core
    s = w.sum()
    if s == 0:
        return dict(fwhm_major_px=0.0, fwhm_minor_px=0.0,
                    fwhm_major_mas=0.0, fwhm_minor_mas=0.0,
                    bpa_deg=0.0)

    mx = (w * xx).sum() / s
    my = (w * yy).sum() / s
    dx, dy = xx - mx, yy - my

    vxx = (w * dx * dx).sum() / s
    vyy = (w * dy * dy).sum() / s
    vxy = (w * dx * dy).sum() / s

    t = vxx + vyy
    d = vxx - vyy
    r = np.hypot(0.5*d, vxy)
    lam_max = max(0.5*(t + 2*r), 0.0)
    lam_min = max(0.5*(t - 2*r), 0.0)

    smaj = np.sqrt(lam_max)
    smin = np.sqrt(lam_min)
    fwhm_major_px = 2.354820045 * smaj
    fwhm_minor_px = 2.354820045 * smin

    # orientation wrt +x, convert to east-of-north
    theta_x = 0.5*np.arctan2(2*vxy, d)
    bpa_eon = (90.0 - np.degrees(theta_x)) % 180.0

    fmaj_mas = fwhm_major_px * float(cell_mas)
    fmin_mas = fwhm_minor_px * float(cell_mas)

    px_beam = float(2.0 * np.pi * smaj * smin)

    return dict(
        fwhm_major_px=float(fwhm_major_px),
        fwhm_minor_px=float(fwhm_minor_px),
        fwhm_major_mas=float(fmaj_mas),
        fwhm_minor_mas=float(fmin_mas),
        bpa_deg=float(bpa_eon),
        pix_per_beam=px_beam,
    )

def fit_gaussian_core(psf_peak1, cell_mas, win=33, frac_low=0.2, frac_high=0.9):
    H, W = psf_peak1.shape
    vc, uc = H//2, W//2
    ext = win//2
    sl = slice(vc-ext, vc+ext+1)
    sc = slice(uc-ext, uc+ext+1)
    cut = psf_peak1[sl, sc].astype(np.float64)

    # coords centered on the peak pixel
    y, x = np.mgrid[-ext:ext+1, -ext:ext+1]
    peak = cut[ext, ext]
    if peak <= 0:
        return measure_beam_core(psf_peak1, cell_mas)

    z = cut / peak
    mask = (z >= frac_low) & (z <= frac_high)
    if mask.sum() < 6:
        return measure_beam_core(psf_peak1, cell_mas)

    X = np.stack([x[mask]**2, 2*x[mask]*y[mask], y[mask]**2], axis=1)
    yv = -2.0 * np.log(z[mask] + 1e-300)  # solve for a,b,c in: z ≈ exp(-0.5*(a x^2+2bxy + c y^2))

    coeffs, *_ = np.linalg.lstsq(X, yv, rcond=None)
    a, b, c = coeffs
    M = np.array([[a, b], [b, c]], dtype=float)

    # ensure positive-definite
    evals, evecs = np.linalg.eigh(M)
    evals = np.maximum(evals, 1e-12)
    sigmas = 1.0 / np.sqrt(evals)                 # px
    fwhm_px = 2.354820045 * sigmas
    # sort so [major, minor]
    order = np.argsort(fwhm_px)[::-1]
    fwhm_major_px, fwhm_minor_px = fwhm_px[order]

    smaj = (fwhm_major_px) / (2.0*np.sqrt(2.0*np.log(2.0)))
    smin = (fwhm_minor_px) / (2.0*np.sqrt(2.0*np.log(2.0)))

    # BPA (east of north)
    vec_major = evecs[:, order[0]]                # eigenvector of smaller curvature = broader axis
    # angle wrt +x:
    theta_x = np.arctan2(vec_major[1], vec_major[0])
    bpa_eon = (90.0 - np.degrees(theta_x)) % 180.0

    return dict(
        fwhm_major_px=float(fwhm_major_px),
        fwhm_minor_px=float(fwhm_minor_px),
        fwhm_major_mas=float(fwhm_major_px*cell_mas),
        fwhm_minor_mas=float(fwhm_minor_px*cell_mas),
        bpa_deg=float(bpa_eon),
        pix_per_beam=float(2.0 * np.pi * smaj * smin),
    )

def beam_from_target(target_fwhm_mas, axial_ratio, keep="geomean"):
    r = float(axial_ratio)
    if not (0 < r <= 1):
        raise ValueError("axial_ratio must be in (0,1].")
    if keep == "geomean":
        bmaj = target_fwhm_mas / np.sqrt(r)
        bmin = target_fwhm_mas * np.sqrt(r)
    elif keep == "major":
        bmaj = target_fwhm_mas
        bmin = r * bmaj
    elif keep == "minor":
        bmin = target_fwhm_mas
        bmaj = bmin / r
    else:
        raise ValueError("keep must be 'geomean', 'major', or 'minor'.")
    return float(bmaj), float(bmin)

# ---------------------------
# uv-plane noise → image
# ---------------------------
def uv_noise_image(mask_counts, gamma=0.8, seed=0):
    H, W = mask_counts.shape
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(H,W)) + 1j * rng.normal(size=(H,W))
    w = (np.asarray(mask_counts, dtype=np.float64) ** float(gamma)) / np.sqrt(2.0)
    z *= w
    n  = np.fft.ifft2(np.fft.ifftshift(z), norm="ortho")
    n  = np.fft.fftshift(np.real(n))
    return n.astype(np.float32)

# ---------------------------
# Header keywords
# ---------------------------
def mas_to_deg(x_mas):
    return float(x_mas) / 3_600_000.0
def beam_keywords_from_psf(psf, cell_mas):
    m = measure_psf_moments(psf)
    fmaj_mas = m["fwhm_major"] * float(cell_mas)
    fmin_mas = m["fwhm_minor"] * float(cell_mas)
    return {"BMAJ": mas_to_deg(fmaj_mas), "BMIN": mas_to_deg(fmin_mas), "BPA": float(m["bpa"])}

# ---------------------------
# The one-call simulator
# ---------------------------
def simulate_from_array(
    *,                          # keyword-only for safety
    clean_img,                  # 2D ndarray, W m^-2 Hz^-1 "truth"
    target_fwhm_mas,            # expected fwhm (mas)
    axial_ratio,                # minor/major
    ant_ENU_m,                  # (N,3) [E,N,U] meters
    freq_GHz,                   # observing frequency
    cell_mas,                   # mas/pixel
    t_hr,                       # total track time (hr)
    dec_deg,                    # target declination (deg)
    lat_deg=SKA_MID_LAT_DEG,    # site latitude
    dt_s=10.0,                  # dump time (s)
    el_min_deg=20.0,            # min elevation cut
    rms_uJy_per_beam=None,      # target map RMS (µJy/beam) — REQUIRED for noise scaling
    peak_snr=None,              # target peak SNR — REQUIRED for absolute scaling
    gamma=0.8,                  # uv weight flattening (natural→uniform)
    blur_sigma_px=1.25,         # uv gridding blur (pixels)
    restore=True,               # use Gaussian restoring beam for model
    return_sum1_psf=True,       # also return a sum=1 PSF for ML
    residual_psd="dirty",
    seed=0,
    preclip: bool = False,
    all_thermal: bool = True,
):
    clean = np.asarray(clean_img, dtype=np.float64)
    H, W = clean.shape

    # 1) Hour angles, uv counts (with small gridding blur)
    H_grid = hour_angles_for_track(dec_deg, lat_deg, t_hr, dt_s=dt_s, el_min_deg=el_min_deg)
    counts = array_to_uv_counts(ant_ENU_m, dec_deg, freq_GHz, H_grid, (H,W), cell_mas, blur_sigma_px=blur_sigma_px)

    # 2) Dirty PSF (peak ~ 1) and its beam metrics
    mask_norm = counts / (counts.max() if counts.max()!=0 else 1.0)
    psf_dirty = psf_from_uv_mask(mask_norm)

    beam_info = fit_gaussian_core(psf_dirty, cell_mas,)
    bpa_uv = beam_info["bpa_deg"]
    BMAJ_tgt_mas, BMIN_tgt_mas = beam_from_target(target_fwhm_mas, axial_ratio)
    BMAJ_tgt_mas, BMIN_tgt_mas = abs(BMAJ_tgt_mas), abs(BMIN_tgt_mas)
    BPA_tgt_deg   = bpa_uv
    # print(beam_info)
    
    rest_target = gaussian_restoring_beam(H, W, BMAJ_tgt_mas, BMIN_tgt_mas, BPA_tgt_deg, cell_mas)
    psf_for_header = rest_target
    rest_peak1 = rest_target
    beam_info = fit_gaussian_core(rest_target, cell_mas,)
    fmaj_mas = abs(beam_info["fwhm_major_mas"])
    fmin_mas = abs(beam_info["fwhm_minor_mas"])
    bpa_eon = beam_info["bpa_deg"]
    pix_per_beam = beam_info["pix_per_beam"]
    # print(beam_info)

    if residual_psd == "restoring":
        model_psf = rest_target
    elif residual_psd == "dirty":
        model_psf = psf_dirty
    else:
        raise ValueError("model_psf_kind must be 'restoring' or 'dirty'")

    model0 = np.fft.irfft2(
        np.fft.rfft2(clean, norm="ortho") *
        np.fft.rfft2(np.fft.ifftshift(model_psf), norm="ortho"),
        s=(H, W), norm="ortho"
    )

    ##-- build residuals --##
    if residual_psd == "dirty":
        noise_raw = uv_noise_image(counts, gamma=gamma, seed=seed+11)
    elif residual_psd == "restoring":
        # white noise → convolve with restoring beam (peak=1) → smoother residuals
        rng = np.random.default_rng(seed+11)
        white = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float64)
        rest_peak1 = (rest_peak1 if restore and rest_peak1 is not None
                      else gaussian_restoring_beam(H, W, fmaj_mas, fmin_mas, bpa_eon, cell_mas))
        # correlate at beam scale
        Wk = np.fft.rfft2(white, norm="ortho")
        Rk = np.fft.rfft2(np.fft.ifftshift(rest_peak1), norm="ortho")
        noise_raw = np.fft.irfft2(Wk * Rk, s=(H, W), norm="ortho")
    else:
        raise ValueError("residual_psd must be 'dirty' or 'restoring'")

    sigma_rms_W = rms_uJy_per_beam * 1e-32
    if preclip:
        model0 = pre_noise_clip_model(model0, sigma_rms_W, method="asinh", s=3.0, k=12.0)    
    # --- scale residuals to RMS, then set peak from measured RMS (unchanged logic) ---
    noise = scale_noise_to_map_rms(noise_raw, rms_uJy_per_beam)
    noise_rms_meas = corner_rms(noise)
    target_peak_W  = float(peak_snr) * noise_rms_meas
    model = model0 * (target_peak_W / (model0.max() or 1.0))

    dirty = model + noise
    

    # 6) Optional sum-1 PSF for ML
    psf_ml_sum1 = None
    if return_sum1_psf:
        if restore and rest_peak1 is not None:
            psf_ml_sum1 = (rest_peak1 / (rest_peak1.sum() or 1.0)).astype(np.float32)
        else:
            # make a Gaussian restoring PSF that matches the dirty beam
            g = gaussian_restoring_beam(H, W, fmaj_mas, fmin_mas, bpa_eon, cell_mas)
            psf_ml_sum1 = (g / (g.sum() or 1.0)).astype(np.float32)

    # 7) Header-like dict
    beam_keys = beam_keywords_from_psf(psf_for_header, cell_mas)
    header = {
        "BMAJ": beam_keys["BMAJ"], "BMIN": beam_keys["BMIN"], "BPA": beam_keys["BPA"],
        "BTYPE": "Intensity", "BUNIT": "W m-2 Hz-1 / beam"
    }

    meta = dict(
        freq_GHz=float(freq_GHz), cell_mas=float(cell_mas),
        dec_deg=float(dec_deg), lat_deg=float(lat_deg),
        t_hr=float(t_hr), dt_s=float(dt_s), el_min_deg=float(el_min_deg),
        rms_uJy_per_beam=float(rms_uJy_per_beam), rms_W_per_beam=float(sigma_rms_W),
        peak_snr=float(peak_snr),
        measured_noise_rms_W=float(noise_rms_meas),
        fitted_fwhm_major_mas=float(fmaj_mas),
        fitted_fwhm_minor_mas=float(fmin_mas),
        bmaj_deg=float(fmaj_mas)/(3600e3),
        bmin_deg=float(fmin_mas)/(3600e3),
        bpa_deg=float(bpa_eon),
        px_per_beam=float(pix_per_beam),
        total_hits=float(counts.sum()),
        mean_hits_per_filled_pixel=float(counts.sum()/max((counts>0).sum(),1)),
    )

    outputs = dict(
        dirty_image=dirty,
        psf_dirty_peak1=psf_dirty.astype(np.float32),
        restoring_beam_peak1=(rest_peak1.astype(np.float32) if rest_peak1 is not None else None),
        psf_ml_sum1=psf_ml_sum1,
        uv_counts=counts.astype(np.float32),
        header=header,
        meta=meta,
        model=model,
    )
    return outputs


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate SKA-Mid dirty images.")
    parser.add_argument("--data_dir", type=str, default="../data/ska_cont_planets_off/", help="Path to directory containing clean images.")
    parser.add_argument("--clean_output_dir", type=str, default="../data/ska_cont_clean_planets_off/", help="Path to directory where clean images will be saved.")
    parser.add_argument("--dirty_output_dir", type=str, default="../data/ska_cont_dirty_planets_off/", help="Path to directory where dirty images will be saved.")
    parser.add_argument("--telescope", type=str, default="ska", choices=["ska", "alma"], help="Telescope to simulate dirty images for.")
    parser.add_argument("--clip_percentile", type=float, default=0.1, help="Percentile to clip clean images to.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for dirty image simulation.")
    parser.add_argument("--enforce_index", type=int, default=0, help="Make certain obsevation (0 = ignore)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument("--run_prefix", default="planet", help="Prefix of dumpfiles")
    parser.add_argument("--cube_path", default="../data/data_cube.dat", help="Path to data cube")
    parser.add_argument("--cfg_path", default="../data/SKA1MID_LOC.cfg", help="Path to config file")
    parser.add_argument("--residual_psd", type=str, default="restoring", choices=["dirty", "restoring"], 
                        help="Residuals (dirty or restoring beam).")
    
    args = parser.parse_args()
    
    which_telescope = args.telescope
    # ENU coordinates of antennas 
    ants = np.loadtxt(args.cfg_path, comments="#", usecols=(0,1,2))
    
    data_path = args.data_dir
    data_list = [x for x in os.listdir(data_path) if ".fits" in x]

    clean_result_dir = args.clean_output_dir
    dirty_result_dir = args.dirty_output_dir
    os.makedirs(clean_result_dir, exist_ok=True)
    os.makedirs(dirty_result_dir, exist_ok=True)
    
    clip_percentile = args.clip_percentile

    pct_1 = 10 # 67, 10
    pct_2 = 25 # 67, 100
    pct_3 = 5 # 67, 1000
    pct_4 = 10 # 34, 10
    pct_5 = 25 # 34, 100
    pct_6 = 25 # 34, 1000

    obs_1_min = 0.
    obs_2_min = pct_1 / 100.
    obs_3_min = pct_2 / 100. + obs_2_min
    obs_4_min = pct_3 / 100. + obs_3_min
    obs_5_min = pct_4 / 100. + obs_4_min
    obs_6_min = pct_5 / 100. + obs_5_min

    np.random.seed(123)
    
    df = pd.read_csv(args.cube_path, 
                    sep="\t", 
                    )

    obs_meta_data = {}

    i = 0
    for run in df.Run.to_numpy():
        
        this_df = df[df["Run"] == run]
        for im in [x for x in data_list if f"{args.run_prefix}{run}_" in x and ".fits" in x]:

            if im != "ska_cont_planet232_00644_ska_clean.fits":
                continue
            with fits.open(f"{data_path}{im}") as hdul:
                clean = hdul[0].data
                if clean.ndim == 3:
                    clean = clean[0]
                clean = clean.squeeze().astype(np.float64)
                hdr0 = hdul[0].header

            # optional dynamic range clipping
            if clip_percentile > 0.:
                lo, hi = np.percentile(clean, [clip_percentile, 100. - clip_percentile])
                clean = np.clip(clean, lo, hi, out=clean)

            # choose beam (arcsec) around your target resolution
            pa_deg      = float(this_df.position_angle.to_numpy()[0])
            dist_pc     = float(this_df.dist.to_numpy()[0])
            beam_frac  = np.random.uniform(low=0.5, high=1.)
            obs_frac  = np.random.uniform(low=0., high=1.)

            cell_mas = abs(hdr0["CDELT1"] * 3600 * 1e3)
            
            enforce_index = args.enforce_index
            if enforce_index == 1:
                obs_frac = obs_1_min
            elif enforce_index == 2:
                obs_frac = obs_2_min
            elif enforce_index == 3:
                obs_frac = obs_3_min
            elif enforce_index == 4:
                obs_frac = obs_4_min
            elif enforce_index == 5:
                obs_frac = obs_5_min
            elif enforce_index == 6:
                obs_frac = obs_6_min

            restore = True
            if obs_frac >= obs_1_min and obs_frac < obs_2_min:
                # 67 mas, 10 hr
                key = ("67mas", 10)
                bmaj_arcsec = 67. * 1e-3
                t_hr = 10. # observations times in hours
                peak_snr = 7.58
                rms = 0.44 # microJy/beam
                sensitivity = 1.2 # microJy/beam
                visibility_noise = 0.28 # mJy
                rms = np.random.uniform(low=0.8 * rms, high=1.2 * rms)
                rms10hr = 0.44
                blur_sigma_px = 2.
                which_obs = 1
                sigma_rms_uJy = np.random.uniform(low=0.8 * peak_snr, high=1.2 * peak_snr)

            elif obs_frac >= obs_2_min and obs_frac < obs_3_min:
                # 67 mas, 100 hr
                key = ("67mas", 100)
                bmaj_arcsec = 67. * 1e-3
                t_hr = 100. # observations times in hours
                peak_snr = 17.17
                rms = 0.14 # microJy/beam
                sensitivity = 1.2 # microJy/beam
                visibility_noise = 0.28 # mJy
                rms = np.random.uniform(low=0.8 * rms, high=1.2 * rms)
                rms10hr = 0.44
                blur_sigma_px = 2.
                which_obs = 2
                sigma_rms_uJy = np.random.uniform(low=0.8 * peak_snr, high=1.2 * peak_snr)
                # continue
            
            elif obs_frac >= obs_3_min and obs_frac < obs_4_min:
                # 67 mas and 1000 hr
                key = ("67mas", 1000)
                bmaj_arcsec = 67. * 1e-3
                rms = 0.14
                t_hr = 1000.
                rms = 0.05 # microJy/beam
                sensitivity = 1.2 # microJy/beam
                visibility_noise = 0.28 # mJy
                peak_snr = 44.75
                snr = np.random.uniform(low=0.8 * peak_snr, high=1.2 * peak_snr)
                rms10hr = 0.44
                blur_sigma_px = 1.5
                which_obs = 3
                sigma_rms_uJy = np.random.uniform(low=0.8 * rms, high=1.2 * rms)

            elif obs_frac >= obs_4_min and obs_frac < obs_5_min:
                # 34 mas and 10 hours
                key = ("34mas", 10)
                bmaj_arcsec = 34. * 1e-3
                t_hr = 10.
                rms = 0.83 # microJy/beam
                peak_snr = 4.90
                sensitivity = 2.4
                visibility_noise = 0.56
                rms10hr = 0.83
                blur_sigma_px = 2.
                which_obs = 4
                snr = np.random.uniform(low=0.8 * peak_snr, high=1.2 * peak_snr)
                sigma_rms_uJy = np.random.uniform(low=0.8 * rms, high=1.2 * rms)
            
            elif obs_frac >= obs_5_min and obs_frac < obs_6_min:
                # 34 mas and 100 hours
                key = ("34mas", 100)
                bmaj_arcsec = 34. * 1e-3
                t_hr = 100.
                rms = 0.26 # microJy/beam
                peak_snr = 7.73
                sensitivity = 2.4
                visibility_noise = 0.56
                rms10hr = 0.83
                blur_sigma_px = 2.
                which_obs = 5
                snr = np.random.uniform(low=0.8 * peak_snr, high=1.2 * peak_snr)
                sigma_rms_uJy = np.random.uniform(low=0.8 * rms, high=1.2 * rms)

            else:
                # 34 mas and 1000 hr
                key = ("34mas", 1000)
                bmaj_arcsec = 34. * 1e-3
                t_hr = 1000.
                peak_snr = 16.8
                rms = 0.08
                sensitivity = 2.4
                rms10hr = 0.83
                visibility_noise = 0.56
                blur_sigma_px = 2.
                which_obs = 6
                snr = np.random.uniform(low=0.8 * peak_snr, high=1.2 * peak_snr)
                sigma_rms_uJy = np.random.uniform(low=0.8 * rms, high=1.2 * rms)

            h_min_deg = 20.
            dec_deg = sample_declinations(1,  h_min_deg=h_min_deg)[0]

            residual_psd = args.residual_psd
            
            dt_s = np.random.uniform(low=10., high=30.)

            p = SKA_PRESETS[key]

            rms10hr = SKA_PRESETS[(key[0], 10)]["rms"]
            peak_snr = p["peak_snr"]
            rms = p["rms"]
            visibility_noise = p["vis"]
            radial_exp = p["radial_exp"]
            uv_points = p["uv_points"]
            # gamma = p[key]["gamma"]
            snr = np.random.uniform(low=0.8 * peak_snr, high=1.2 * peak_snr)
            sigma_rms_uJy = np.random.uniform(low=0.8 * rms, high=1.2 * rms)
            

            freq_GHz = 12.5
            
            gamma = 1.
            axial_ratio = 1.
            axial_ratio_min = 0.5
            axial_ratio = np.random.uniform(low=axial_ratio_min, high=1.)
            
            obs_meta_data[im] = {
                "axial_ratio": axial_ratio,
                "peak_snr": peak_snr,
                "obs": which_obs,
                "rms": rms,
                "rms10hr": rms10hr,
                "visibility_noise": visibility_noise,
                "radial_exp": radial_exp,
                "uv_points": uv_points,
                "gamma": gamma,
                "snr": snr,
                "sigma_rms_uJy": sigma_rms_uJy,
            }

            outputs = simulate_from_array(
                clean_img=clean,                  # 2D ndarray, W m^-2 Hz^-1 "truth"
                target_fwhm_mas=bmaj_arcsec*1e3,  
                axial_ratio=axial_ratio,
                ant_ENU_m=ants,                  # (N,3) [E,N,U] meters
                freq_GHz=12.5,                   # observing frequency
                cell_mas=cell_mas,                   # mas/pixel
                t_hr=t_hr,                       # total track time (hr)
                dec_deg=dec_deg,                    # target declination (deg)
                lat_deg=SKA_MID_LAT_DEG,    # site latitude
                dt_s=dt_s,                  # dump time (s)
                el_min_deg=h_min_deg,            # min elevation cut
                rms_uJy_per_beam=sigma_rms_uJy,      # target map RMS (µJy/beam) — REQUIRED for noise scaling
                peak_snr=snr,              # target peak SNR — REQUIRED for absolute scaling   
                restore=restore,               # use Gaussian restoring beam for model
                return_sum1_psf=False,       # also return a sum=1 PSF for ML
                seed=123,
                residual_psd=residual_psd,
                blur_sigma_px=blur_sigma_px, # uv gridding blur (pixels)
                gamma=gamma,               # uv weight flattening (natural→uniform)
                preclip=clip_percentile == 0.,
            )

            dirty = outputs["dirty_image"]
            psf = outputs["psf_dirty_peak1"]
            meta = outputs["meta"]
            uv_counts = outputs["uv_counts"]
                    
            # --- write FITS: primary = dirty image; EXT1 = PSF kernel ---
            phdu = fits.PrimaryHDU(dirty)
            hdr = phdu.header
            hdr["NPLANETS"] = (int(this_df.N_planet.to_numpy()[0]) if "N_planet" in this_df.columns else 0, 
                               "number of planets in disc")
            hdr["DIST"]     = (dist_pc, "distance in parsecs")
            hdr["AU_PP"]    = (1., "au per pixel")                 # au per pixel
            hdr["BMAJ"]     = (abs(meta["fitted_fwhm_major_mas"] / 3600e3), "beam major axis in degrees")
            hdr["BMIN"]     = (abs(meta["fitted_fwhm_minor_mas"] / 3600e3), "beam minor axis in degrees")
            hdr["BPA"]      = (meta["bpa_deg"], "beam position angle in degrees")
            hdr["SNR"]      = (meta["peak_snr"], "peak signal to noise ratio")
            hdr["RMS"] = (meta["measured_noise_rms_W"], "noise RMS in CUNIT1")
            hdr["MAS_PIX"] = cell_mas
            hdr["BRMS_uJY"] = (meta["rms_uJy_per_beam"], "RMS per beam (uJy)")
            hdr["BRMS_W"] = (meta["rms_W_per_beam"], "RMS per beam (CUNIT1)")
            hdr["t_EXP"] = (t_hr, "total exposure time (hours)")
            hdr["dt"] = (dt_s, "single exposure time (seconds)")
            hdr["H_MIN"] = (h_min_deg, "minimum elevation cut (degrees)")
            hdr["DEC"] = (dec_deg, "declination (degrees)")
            hdr["UV_CNTS"] = np.sum(uv_counts)
            hdr["TOT_HITS"] = meta["total_hits"]
            hdr["BEAM_PIX"] = (meta["px_per_beam"], "pixels per beam")
            hdr["FREQ"] = (freq_GHz * 1e9, "observation frequency (Hz)")
            hdr["BUNIT"] = "W m-2 Hz-1 beam-1"
            hdr["RES_PSD"] = residual_psd
            hdr["PIX_BLUR"] = blur_sigma_px
            hdr["GAMMA"]   = gamma
            hdr["RATIO"]   = (axial_ratio, "Axial ratio between beam axes")
            hdr["RES_PSD"]   = (residual_psd, "PSD noising method")
            hdr["REST"]   = (restore, "Use restoring beam")
            hdr["INSTR"]  = (which_telescope, "Instrument(s) used")
            hdr["OBS_IDX"] = which_obs
            for key in ["CDELT1", "CDELT2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRPIX1", "CRPIX2"]:
                hdr[key] = hdr0[key]

            psf_hdu = fits.ImageHDU(psf.astype(np.float32), name="PSF")
            if restore:
                rest_hdu = fits.ImageHDU(outputs["restoring_beam_peak1"].astype(np.float32), name="rest_beam")
            for key, value in hdr.items():
                psf_hdu.header[key] = value
                phdu.header[key] = value
                if restore:
                    rest_hdu.header[key] = value
            hdul_out = fits.HDUList([phdu, psf_hdu] if not restore else [phdu, psf_hdu, rest_hdu])

            if enforce_index > 0:
                which_telescope = f"{which_telescope}_{enforce_index}"
            
            dirty_name = f"{dirty_result_dir}{im.split('.fits')[0]}_{which_telescope}_dirty.fits"
            hdul_out.writeto(dirty_name, overwrite=True)

            fits.writeto(f"{clean_result_dir}{im.split('.fits')[0]}_{which_telescope}_clean.fits", clean, header=hdr, overwrite=True)
            i += 1
            print(f"Done with {i}", end='\r')
            if np.max(psf) > 2 or np.min(psf) < -2:
                print(f"PSF bad range: {im}")
                
    with open(f'{dirty_result_dir}obs_meta_data.yml', 'w') as outfile:
        yaml.dump(obs_meta_data, outfile, default_flow_style=False)
                
    print("\ndone")