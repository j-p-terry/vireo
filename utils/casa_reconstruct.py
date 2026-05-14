#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for CASA multiscale CLEAN baseline for SKA-Mid-like synthetic images,
with optional auto second-pass based on residual peak vs sigma inside the mask.

Logging:
- <root>.params.json (per-image, machine-readable)
- casa_runs.tsv (append-only audit table)
"""

import os
import math
import time
import json
import numpy as np

# ---- CASA imports (tasks + tools) ----
from casatasks import (
    importfits, exportfits, deconvolve, imsmooth, immath, imhead, imstat, imfit
)
from casatools import image as iatool, quanta, logsink
ia = iatool()
qa = quanta()
casalog = logsink()

# -----------------------------
# User parameters
# -----------------------------
gain         = 0.05
snr_stop     = 4.0          # SNR>snr_stop seeds the mask
sigma_stop   = 1.2          # CLEAN threshold in units of sigma (Jy); pass 1
niter        = 50000
smooth_frac  = 0.5          # mask-grow kernel = smooth_frac × beam
post_smooth  = 1.00         # optional cosmetic smoothing factor on restored image

# Second pass controls (pick one of the following modes)
deconvolve_again      = False     # manual on/off (forces a second pass if True)
auto_second_pass      = True      # auto rule: run pass-2 only if residual_peak/σ2 >= threshold
auto_second_threshold = 3.5       # default T in rule above
second_pass_gain      = 0.03
second_pass_snr_boost = 1.2       # rebuild mask with slightly stricter SNR cut
verbose               = False

# IO (defaults; modify accordingly)
data_dir = "../data/ska_cont_dirty/"
save_dir = "../data/all_casa_clean/"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Small helpers
# -----------------------------
def vprint(*a, **k):
    if verbose: print(*a, **k)

def ensure_4d(imname: str) -> None:
    """Make an image 4D (RA, DEC, STOKES, FREQ) so deconvolve behaves."""
    ia.open(imname)
    shp = ia.shape()
    ia.close()
    if len(shp) == 2:
        tmp = imname + ".4d"
        ia.open(imname)
        ia.adddegaxes(outfile=tmp, stokes='I', spectral=True)  # adds 1x1 axes
        ia.close()
        os.system(f"rm -rf {imname}")
        os.system(f"mv {tmp} {imname}")

def sigma_clip_stats(arr, nsig=3.0, iters=5):
    x = np.asarray(arr, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    m = np.median(x)
    s = 1.4826*np.median(np.abs(x - m))
    for _ in range(iters):
        if not np.isfinite(s) or s <= 0:
            break
        sel = np.abs(x - m) < nsig*s
        if sel.sum() == x.size:
            break
        x = x[sel]
        m = np.median(x)
        s = 1.4826*np.median(np.abs(x - m))
    return float(m), float(s)

def estimate_sigma_from_corners(imname: str, frac=0.05):
    """RMS from four corner boxes with sigma-clipping (Jy/beam)."""
    ia.open(imname)
    nx, ny = ia.shape()[0], ia.shape()[1]
    ia.close()
    m = int(round(min(nx, ny)*frac))
    boxes = [
        (0,     0,     m-1,   m-1),
        (nx-m,  0,     nx-1,  m-1),
        (0,     ny-m,  m-1,   ny-1),
        (nx-m,  ny-m,  nx-1,  ny-1),
    ]
    vals = []
    for (x0,y0,x1,y1) in boxes:
        b = f"{x0},{y0},{x1},{y1}"
        stb = imstat(imname, box=b)
        if 'rms' in stb:
            vals.append(stb['rms'][0])
    med, _ = sigma_clip_stats(vals, nsig=2.5, iters=3)
    return float(med)

def safe_psf_fit(psf_im: str, resid_im: str):
    """Fit a PSF core Gaussian; fallback to header beam."""
    pst = imstat(psf_im)
    if 'max' not in pst or not np.isfinite(pst['max'][0]) or float(pst['max'][0]) <= 0.0:
        hdpsf = imhead(psf_im, mode='list')
        return (qa.convert(hdpsf['beammajor'],'arcsec')['value'],
                qa.convert(hdpsf['beamminor'],'arcsec')['value'],
                qa.convert(hdpsf['beampa'],  'deg')['value'])
    hdres = imhead(resid_im, mode='list')
    nx, ny = hdres['shape'][0], hdres['shape'][1]
    half = int(max(8, min(25, nx//8, ny//8)))
    cx, cy = nx//2, ny//2
    box = f"{max(0,cx-half)},{max(0,cy-half)},{min(nx-1,cx+half)},{min(ny-1,cy+half)}"
    fit = imfit(imagename=psf_im, box=box)
    if 'results' in fit and any(k.startswith('component') for k in fit['results']):
        k = sorted([k for k in fit['results'] if k.startswith('component')])[0]
        comp = fit['results'][k]['shape']
        return (qa.convert(comp['majoraxis'],   'arcsec')['value'],
                qa.convert(comp['minoraxis'],   'arcsec')['value'],
                qa.convert(comp['positionangle'],'deg')['value'])
    hdpsf = imhead(psf_im, mode='list')
    return (qa.convert(hdpsf['beammajor'],'arcsec')['value'],
            qa.convert(hdpsf['beamminor'],'arcsec')['value'],
            qa.convert(hdpsf['beampa'],  'deg')['value'])

def build_snr_mask(root: str, sigma_Jy: float, bmaj_arcsec: float, bmin_arcsec: float):
    """Create SNR map and grown mask; CASA will auto-detect <root>.mask."""
    immath(imagename=[f"{root}.residual"], expr=f"IM0/{sigma_Jy}", outfile=f"{root}.snr")
    immath(imagename=[f"{root}.snr"], expr=f"iif(IM0>{snr_stop}, 1.0, 0.0)", outfile=f"{root}.mask0")
    imsmooth(imagename=f"{root}.mask0", outfile=f"{root}.mask0.sm",
             kernel='gauss',
             major=f"{smooth_frac*bmaj_arcsec}arcsec",
             minor=f"{smooth_frac*bmin_arcsec}arcsec", pa='0deg',
             overwrite=True)
    immath(imagename=[f"{root}.mask0.sm"], expr="iif(IM0>0.3,1.0,0.0)", outfile=f"{root}.mask")
    os.system(f"rm -rf {root}.mask0 {root}.mask0.sm")

def choose_scales(beam_px: float, max_scale: int, pass2: bool=False):
    """Integer multiscale sizes in pixels, tied to pixels-per-beam, within max_scale."""
    seeds = [0, beam_px/4, beam_px/2, beam_px, 2*beam_px, 4*beam_px] + ([] if pass2 else [8*beam_px])
    scales = sorted({int(round(s)) for s in seeds if int(round(s)) >= 0})
    scales = [s for s in scales if (s == 0 or s < max_scale)]
    return scales if scales else [0, 1]

def convert_to_jy_per_beam(imname: str, bunit_orig: str):
    """Convert image to Jy/beam if it's in SI; return True if converted."""
    if "W" in bunit_orig:
        ia.open(imname)
        dat = ia.getchunk()
        ia.putchunk(dat/1e-26)           # SI -> Jy
        ia.setbrightnessunit('Jy/beam')
        ia.close()
        return True
    return False

# --------- Logging helpers ----------
def _try_get_casa_version():
    v = None
    try:
        v = ia.version()
    except Exception:
        pass
    return v if v is not None else "unknown"

def gather_run_params(root, bunit_orig, cell_arcsec, bmaj_arcsec, bmin_arcsec, bpa_deg,
                      beam_px, nx, ny, sigma_Jy, snr_stop, smooth_frac,
                      gain, sigma_stop, niter_used, scales,
                      pass2_mode, auto_threshold, deconvolver="multiscale"):
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image": root,
        "casa_version": _try_get_casa_version(),
        "units_original": bunit_orig,
        "working_units": "Jy/beam",
        "export_units": "W m^-2 Hz^-1 beam^-1",
        "pixel_scale_arcsec": float(cell_arcsec),
        "shape_xy": [int(nx), int(ny)],
        "beam": {
            "bmaj_arcsec": float(bmaj_arcsec),
            "bmin_arcsec": float(bmin_arcsec),
            "bpa_deg": float(bpa_deg),
            "px_per_beam_geom": float(beam_px)
        },
        "noise_sigma_Jy_per_beam": float(sigma_Jy),
        "mask": {
            "snr_stop": float(snr_stop),
            "grow_kernel_frac_of_beam": float(smooth_frac),
            "rethreshold_value": 0.3,
            "filename": f"{root}.mask"
        },
        "deconvolution": {
            "task": "deconvolve",
            "deconvolver": deconvolver,
            "scales_pixels": list(map(int, scales)),
            "gain": float(gain),
            "threshold_sigma": float(sigma_stop),
            "threshold_abs_Jy": float(sigma_stop*sigma_Jy),
            "niter": int(niter_used),
            "second_pass_mode": pass2_mode,  # 'off' | 'manual' | 'auto'
            "auto_second_threshold_sigma": float(auto_threshold) if pass2_mode == "auto" else None,
            "second_pass_run": False,
            "second_pass_reason": None
        },
        "outcomes": {
            "sigma_corners_after_Jy_per_beam": None,
            "residual_peak_inside_mask_Jy_per_beam": None,
            "residual_peak_over_sigma": None,
            "restored_post_smooth_factor": None,
            "output_files": []
        }
    }

def log_params(root, params, save_dir="./casa_clean"):
    """Write JSON next to outputs and append a TSV row for quick audits."""
    # JSON
    json_path = os.path.join(save_dir, f"{root}.params.json")
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)

    # TSV (append)
    tsv_path = os.path.join(save_dir, "casa_runs.tsv")
    hdr = ("timestamp\timage\tcasa\tcell_arcsec\tbmaj\tbmin\tbpa\tpx_per_beam\t"
           "nx\tny\tsigma_Jy\tsnr_stop\tsmooth_frac\tgain\tthr_sigma\tniter\t"
           "scales\tsecond_mode\tauto_thr\tp2_run\tp2_reason\tsigma_after_Jy\t"
           "resid_peak_Jy\tpeak_over_sigma\tpost_smooth\n")
    if not os.path.exists(tsv_path):
        with open(tsv_path, "w") as f: f.write(hdr)
    with open(tsv_path, "a") as f:
        f.write("{ts}\t{img}\t{cv}\t{cell:.6g}\t{bmaj:.6g}\t{bmin:.6g}\t{bpa:.3g}\t{ppb:.3g}\t"
                "{nx}\t{ny}\t{sig:.6g}\t{snr:.3g}\t{sf:.3g}\t{g:.3g}\t{thr:.3g}\t{nit}\t"
                "{sc}\t{mode}\t{athr}\t{p2}\t{why}\t{sig2}\t{rpk}\t{ros}\t{ps}\n".format(
                    ts=params["timestamp_utc"], img=root,
                    cv=str(params["casa_version"]), cell=params["pixel_scale_arcsec"],
                    bmaj=params["beam"]["bmaj_arcsec"], bmin=params["beam"]["bmin_arcsec"],
                    bpa=params["beam"]["bpa_deg"], ppb=params["beam"]["px_per_beam_geom"],
                    nx=params["shape_xy"][0], ny=params["shape_xy"][1],
                    sig=params["noise_sigma_Jy_per_beam"],
                    snr=params["mask"]["snr_stop"], sf=params["mask"]["grow_kernel_frac_of_beam"],
                    g=params["deconvolution"]["gain"], thr=params["deconvolution"]["threshold_sigma"],
                    nit=params["deconvolution"]["niter"],
                    sc=",".join(map(str, params["deconvolution"]["scales_pixels"])),
                    mode=params["deconvolution"]["second_pass_mode"],
                    athr=(params["deconvolution"]["auto_second_threshold_sigma"]
                          if params["deconvolution"]["second_pass_mode"] == "auto" else ""),
                    p2=int(params["deconvolution"]["second_pass_run"]),
                    why=(params["deconvolution"]["second_pass_reason"] or ""),
                    sig2=(params["outcomes"]["sigma_corners_after_Jy_per_beam"]
                          if params["outcomes"]["sigma_corners_after_Jy_per_beam"] is not None else ""),
                    rpk=(params["outcomes"]["residual_peak_inside_mask_Jy_per_beam"]
                          if params["outcomes"]["residual_peak_inside_mask_Jy_per_beam"] is not None else ""),
                    ros=(params["outcomes"]["residual_peak_over_sigma"]
                          if params["outcomes"]["residual_peak_over_sigma"] is not None else ""),
                    ps=(params["outcomes"]["restored_post_smooth_factor"]
                        if params["outcomes"]["restored_post_smooth_factor"] is not None else "")
                ))

# --------- Decision metric for auto pass-2 ----------
def residual_peak_inside_mask(root: str) -> float:
    """Return residual peak [Jy/beam] *inside* the binary mask (mask>0.5)."""
    # read residual
    ia.open(f"{root}.residual"); res = ia.getchunk(); ia.close()
    # read mask (numeric 0/1)
    ia.open(f"{root}.mask"); m = ia.getchunk(); ia.close()
    mm = (m > 0.5) & np.isfinite(res)
    if not mm.any():
        return 0.0
    return float(np.nanmax(res[mm]))

# -----------------------------
# Main loop
# -----------------------------
fits_files = sorted([x for x in os.listdir(data_dir) if x.endswith(".fits")])
done = 0

for mef in fits_files:
    root = f"{os.path.splitext(mef)[0]}_casa"

    if os.path.exists(os.path.join(save_dir, f"{root}.restored.fits")):
        print(f"Already done {mef}")
        done += 1
        continue

    # bring the MEF locally
    os.system(f"cp {os.path.join(data_dir, mef)} .")
    vprint(f"Processing {mef}")

    # 1) Import MEF HDUs into CASA images (assumed order: residual=HDU0, psf=HDU1)
    importfits(fitsimage=mef, imagename=f"{root}.residual", whichhdu=0, overwrite=True)
    importfits(fitsimage=mef, imagename=f"{root}.psf",      whichhdu=1, overwrite=True)

    # sanity check: PSF positive max
    pst = imstat(f"{root}.psf")
    if 'max' not in pst or not np.isfinite(pst['max'][0]) or pst['max'][0] <= 0:
        raise RuntimeError(f"PSF import looks invalid for {mef}; check HDU mapping.")

    # original units
    ia.open(f"{root}.residual"); bunit_orig = ia.brightnessunit(); ia.close()

    # 2) Ensure 4D shape
    ensure_4d(f"{root}.residual")
    ensure_4d(f"{root}.psf")

    # 3) Zero model if missing + sanitize NaNs
    if not os.path.exists(f"{root}.model"):
        immath(imagename=[f"{root}.residual"], expr="0*IM0", outfile=f"{root}.model")
    for suf in ("residual","psf"):
        ia.open(f"{root}.{suf}")
        dat = ia.getchunk()
        bad = ~np.isfinite(dat)
        if bad.any():
            dat[bad] = 0.0 if suf == "psf" else np.nanmedian(dat[np.isfinite(dat)])
            ia.putchunk(dat)
        ia.close()

    # 4) Convert to Jy/beam for thresholding (if needed)
    _ = convert_to_jy_per_beam(f"{root}.residual", bunit_orig)
    _ = convert_to_jy_per_beam(f"{root}.model",    bunit_orig)

    # 5) Pixel scale and PSF fit -> restoring beam + scales
    hd = imhead(f"{root}.residual", mode='list')
    cell_arcsec = abs(qa.convert(hd['cdelt2'],'arcsec')['value'])
    bmaj_arcsec, bmin_arcsec, bpa_deg = safe_psf_fit(f"{root}.psf", f"{root}.residual")
    bgeom_arcsec = math.sqrt(max(bmaj_arcsec,0.0)*max(bmin_arcsec,0.0))
    beam_px = bgeom_arcsec / max(cell_arcsec, 1e-12)
    if beam_px < 3.0:
        casalog.post("WARNING: Beam is poorly sampled (<3 px/beam). Consider regridding to smaller cell.", "WARN")
    nx, ny = hd['shape'][0], hd['shape'][1]
    max_scale = int(0.35 * min(nx, ny))

    # 6) Estimate sigma from corners (Jy/beam)
    sigma_Jy = estimate_sigma_from_corners(f"{root}.residual")
    vprint(f"Corner sigma ≈ {sigma_Jy:.3e} Jy/beam")

    # 7) Build SNR mask (CASA will auto-detect <root>.mask)
    build_snr_mask(root, sigma_Jy, bmaj_arcsec, bmin_arcsec)

    # 8) First multiscale pass
    scales = choose_scales(beam_px, max_scale, pass2=False)
    niter_used = min(20000, niter//3) if not (deconvolve_again or auto_second_pass) else niter//2
    vprint("Pass-1 multiscale scales (px):", scales)

    # Multiscale deconvole with mask auto-detection
    deconvolve(imagename=root,
               deconvolver='multiscale',
               scales=scales,
               niter=niter_used,
               gain=gain,
               threshold=f'{sigma_stop * sigma_Jy}Jy')

    # --- Log params before deciding pass-2 ---
    pass2_mode = ("manual" if deconvolve_again else ("auto" if auto_second_pass else "off"))
    params = gather_run_params(
        root=root,
        bunit_orig=bunit_orig,
        cell_arcsec=cell_arcsec,
        bmaj_arcsec=bmaj_arcsec, bmin_arcsec=bmin_arcsec, bpa_deg=bpa_deg,
        beam_px=beam_px, nx=nx, ny=ny,
        sigma_Jy=sigma_Jy, snr_stop=snr_stop, smooth_frac=smooth_frac,
        gain=gain, sigma_stop=sigma_stop, niter_used=niter_used,
        scales=scales, pass2_mode=pass2_mode, auto_threshold=auto_second_threshold
    )
    log_params(root, params, save_dir=save_dir)

    # 9) Decide and (optionally) run second pass
    run_pass2 = False
    reason = None

    sigma2 = estimate_sigma_from_corners(f"{root}.residual")
    r_peak = residual_peak_inside_mask(root)
    peak_over_sigma = (r_peak / max(sigma2, 1e-12)) if sigma2 > 0 else 0.0

    if deconvolve_again:
        run_pass2 = True
        reason = "manual=True"
    elif auto_second_pass:
        if peak_over_sigma >= auto_second_threshold:
            run_pass2 = True
            reason = f"auto: peak/sigma={peak_over_sigma:.2f} >= {auto_second_threshold:.2f}"
        else:
            reason = f"auto: peak/sigma={peak_over_sigma:.2f} < {auto_second_threshold:.2f}"

    if run_pass2:
        # Optionally rebuild mask at slightly stricter SNR
        if second_pass_snr_boost and sigma2 > 0:
            for t in ('.snr','.mask0','.mask','.mask0.sm'):
                os.system(f"rm -rf {root}{t}")
            immath(imagename=[f"{root}.residual"], expr=f"IM0/{sigma2}", outfile=f"{root}.snr")
            immath(imagename=[f"{root}.snr"], expr=f"iif(IM0>{second_pass_snr_boost*snr_stop}, 1.0, 0.0)", outfile=f"{root}.mask0")
            imsmooth(imagename=f"{root}.mask0", outfile=f"{root}.mask0.sm",
                     kernel='gauss',
                     major=f"{smooth_frac*bmaj_arcsec}arcsec",
                     minor=f"{smooth_frac*bmin_arcsec}arcsec", pa='0deg',
                     overwrite=True)
            immath(imagename=[f"{root}.mask0.sm"], expr="iif(IM0>0.3,1.0,0.0)", outfile=f"{root}.mask")
            os.system(f"rm -rf {root}.mask0 {root}.mask0.sm")

        scales2 = choose_scales(beam_px, max_scale, pass2=True)
        vprint("Pass-2 multiscale scales (px):", scales2)
        deconvolve(imagename=root,
                   deconvolver='multiscale',
                   scales=scales2,
                   niter=min(25000, niter//2),
                   gain=second_pass_gain,
                   threshold=f'{min(1.2*sigma2, sigma_stop*sigma_Jy)}Jy')
        # Update diagnostics after pass-2
        sigma2 = estimate_sigma_from_corners(f"{root}.residual")
        r_peak = residual_peak_inside_mask(root)
        peak_over_sigma = (r_peak / max(sigma2, 1e-12)) if sigma2 > 0 else 0.0

    # 10) Restore: model ⊗ beam + residual
    imsmooth(imagename=f"{root}.model", outfile=f"{root}.modelconv",
             kernel='gauss',
             major=f"{bmaj_arcsec}arcsec", minor=f"{bmin_arcsec}arcsec", pa=f"{bpa_deg}deg",
             overwrite=True)
    immath(imagename=[f"{root}.modelconv", f"{root}.residual"],
           expr="IM0+IM1", outfile=f"{root}.restored")

    # 11) Optional cosmetic smoothing
    if post_smooth > 1.0:
        imsmooth(imagename=f"{root}.restored", outfile=f"{root}.restored_smooth",
                 kernel='gauss',
                 major=f"{post_smooth*bmaj_arcsec}arcsec",
                 minor=f"{post_smooth*bmin_arcsec}arcsec",
                 pa=f"{bpa_deg}deg", overwrite=True)

    # 12) Fill outcomes & log
    params["deconvolution"]["second_pass_run"] = bool(run_pass2)
    params["deconvolution"]["second_pass_reason"] = reason
    params["outcomes"]["sigma_corners_after_Jy_per_beam"] = float(sigma2)
    params["outcomes"]["residual_peak_inside_mask_Jy_per_beam"] = float(r_peak)
    params["outcomes"]["residual_peak_over_sigma"] = float(peak_over_sigma)
    params["outcomes"]["restored_post_smooth_factor"] = float(post_smooth)
    params["outcomes"]["output_files"] = [f for f in os.listdir(".") if f.startswith(root) and f.endswith(".fits")]
    log_params(root, params, save_dir=save_dir)

    # 13) Export to FITS in SI (mask/snr left dimensionless)
    def _export_all():
        prods = ("model","residual","restored", "snr", "mask", "restored_smooth")
        for suf in prods:
            if post_smooth <= 1.0 and suf == "restored_smooth":
                continue
            path = f"{root}.{suf}"
            if not os.path.exists(path): 
                continue
            ia.open(path); dat = ia.getchunk()
            if suf not in ["snr","mask"]:
                ia.putchunk(dat*1e-26)  # Jy -> SI
            ia.setbrightnessunit(bunit_orig)
            ia.close()
            exportfits(imagename=path, fitsimage=f"{root}.{suf}.fits", overwrite=True)
        final = f"{root}.restored.fits"
        if post_smooth > 1.0 and os.path.exists(f"{root}.restored_smooth.fits"):
            os.system(f"mv {root}.restored_smooth.fits {final}")
        os.system(f"mv {final} {save_dir}")
    _export_all()

    # 14) Cleanup local intermediates
    for file in os.listdir():
        if root in file or "imstat" in file:
            os.system(f"rm -rf {file}")
        if file == mef:
            os.system(f"rm -rf {file}")

    done += 1
    print(f"Done with {done} / {len(fits_files)}: {root}.restored.fits", end="\r")

print(f"\nAll done. Wrote outputs + logs to: {save_dir}")
