import argparse
import os
import random
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
import sys
sys.path.insert(0, "./utils/")
from data_utils import make_all_paired_datasets
from model_utils import DelayedEarlyStopping
from vireo import VIREO

# fix determinism for CUDA and ms/ssim
torch.use_deterministic_algorithms(True, warn_only=True)

def set_seed(seed: int = 123) -> None:
    """
    Set global random seeds for reproducible experiments.

    Args:
        seed (int): Random seed to use across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(accelerator: str = "auto") -> torch.device:
    """
    Intelligently select computational device.
    """
    if accelerator == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device_map = {
        "mps": torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }

    if not device_map.get(accelerator, False):
        print(f"Warning: {accelerator} not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(accelerator)

def get_git_info() -> dict[str, str | None]:
    """
    Retrieve git repository information.

    Returns:
        Dict containing git commit hash and branch
    """
    try:
        git_path = shutil.which("git")

        if not git_path:
            print("Error: Git executable not found.")
            return {"commit": None, "branch": None}

        commit = subprocess.check_output(  # noqa:S603
            [git_path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,  # Suppress error output
            timeout=3,  # Add a timeout to prevent hanging
        ).strip()

        branch = subprocess.check_output(  # noqa:S603
            [git_path, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,  # Suppress error output
            timeout=3,  # Add a timeout to prevent hanging
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # More comprehensive error handling
        print("Error: Unable to retrieve git information.")
        return {"commit": None, "branch": None}

    return {"commit": commit, "branch": branch}


def create_wandb_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Create comprehensive configuration for experiment tracking.

    Args:
        args (argparse.Namespace): Parsed command-line arguments

    Returns:
        Dict of configuration parameters
    """
    git_info = get_git_info()
    return {
        **vars(args),
        "pytorch_version": torch.__version__,
        "pl_version": pl.__version__,
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
    }
    
def get_dataloaders(args: argparse.Namespace,):
    
    device = get_device(args.accelerator)
    
    # Load dataset
    try:
        print("Loading data")
        all_data = make_all_paired_datasets(data_path=args.data_dir, 
                                    train_split=args.train_split, 
                                    val_split=args.val_split, 
                                    normalize=args.normalize,
                                    clip=bool(args.clip),
                                    dirty_clip_percentile=args.dirty_clip_percentile,
                                    clean_clip_percentile=args.clean_clip_percentile,
                                    clean_folder=args.clean_folder,
                                    dirty_folder=args.dirty_folder,
                                    planet_path=args.planet_path,
                                    )

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        raise
    # num_workers = 4 if device.type == "mps" else args.num_workers
    num_workers = 4
    print("Making data loaders")
    train_loader = DataLoader(
        all_data["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,  # MPS works better with 0 workers
        pin_memory=device.type != "mps",  # Disable pin_memory for MPS
        persistent_workers=True, 
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        all_data["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,  # MPS works better with 0 workers
        pin_memory=device.type != "mps",  # Disable pin_memory for MPS
        persistent_workers=True, 
        prefetch_factor=2,
    )
    
    test_loader = DataLoader(
        all_data["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,  # MPS works better with 0 workers
        pin_memory=device.type != "mps",  # Disable pin_memory for MPS
        persistent_workers=True, 
        prefetch_factor=2,
    )
    
    return train_loader, val_loader, test_loader


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns:
        argparse.Namespace of validated arguments
    """
    parser = argparse.ArgumentParser(
        prog="VIREO",
        description="PSF-aware UNet for denoising continuum data.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./vireo_checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--final_dir",
        type=str,
        default="./vireo_final",
        help="Directory to save final model",
    )
    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    parser.add_argument(
        "--data_dir", type=str, default="./data/", help="Data directory"
    )
    parser.add_argument("--clean_folder", type=str, default="ska_cont_clean_planets_off", help="Clean data directory")
    parser.add_argument("--dirty_folder", type=str, default="ska_cont_dirty_planets_off", help="Dirty data directory")
    parser.add_argument("--planet_path", type=str, default="./data/data_cube.dat", help="Run metadata file path")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--save_names", type=int, default=1, help="Save names of data")
    parser.add_argument("--clip", type=int, default=0, help="Clip data to percentile")
    parser.add_argument("--clean_clip_percentile", type=float, default=100., help="Clip clean data to this percentile")
    parser.add_argument("--dirty_clip_percentile", type=float, default=100., help="Clip dirty data to this percentile")
    parser.add_argument("--normalize", type=str, default="tanh_range", help="Normalization method")

    # Model architecture
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--layers_per_block", type=int, default=2, help="Layers per UNet block")
    parser.add_argument("--max_capacity", type=int, default=128, help="Maximum CNN channels")
    parser.add_argument("--latent_channels", type=int, default=3, help="Number of channels in latent space")
    parser.add_argument("--latent_scale", type=float, default=1., help="Scale of latent space")
    parser.add_argument("--ctx_ch", type=int, default=128, help="Context channels")
    parser.add_argument("--ctx_capacity", type=int, default=64, help="Context capacity")
    parser.add_argument("--adam_eps", type=float, default=1e-7, help="Adam epsilon")
    parser.add_argument("--pct_start", type=float, default=0.1, help="Percentage of training to start learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--dropblock_prob", type=float, default=0.25, help="Dropblock rate")
    parser.add_argument("--dropblock_size", type=int, default=7, help="Dropblock size")
    parser.add_argument("--skip_drop_prob", type=float, default=0.5, help="Dropout rate for skip connections")
    parser.add_argument("--skip_dropout_noise", type=float, default=0.1, help="Dropout amplitude for skip connections")
    parser.add_argument("--recon_loss_type", type=str, default="ms_ssim", help="Sim loss type")
    parser.add_argument("--use_inception", type=int, default=1, help="Add inception layers")
    parser.add_argument("--n_patches", type=int, default=5, help="Number of patches")
    parser.add_argument("--clamp", type=int, default=1, help="Clamp values in network")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value 0 = ignore")
    parser.add_argument("--use_specnorm", type=int, default=0, help="Use spectral norm")
    parser.add_argument("--use_groupnorm", type=int, default=0, help="Use group norm")
    parser.add_argument("--use_softplus", type=int, default=0, help="Use softplus")
    parser.add_argument("--schedule_beta", type=int, default=1, help="Schedule beta")
    parser.add_argument("--zero_skips", type=int, default=1, help="Zero skip connections")
    parser.add_argument("--add_skip_noise", type=int, default=0, help="Add skip noise")
    parser.add_argument("--skip_noise", type=float, default=0.1, help="Skip noise amplitude")
    parser.add_argument("--schedule_skip_drop", type=int, default=1, help="Schedule skip drop")
    parser.add_argument("--window_size", type=int, default=0, help="Window size for self-attention (0 = global)")
    parser.add_argument("--scale_out", type=float, default=1., help="Scale output")
    parser.add_argument("--final_activation", type=str, default="asinh", help="Add final activation (asinh, softclip, tanh, identity)")
    parser.add_argument("--asinh_k", type=float, default=1., help="Asinh k")
    parser.add_argument("--softclip_a", type=float, default=2., help="Softclip a")
    parser.add_argument("--laplace_weight", type=float, default=0.01, help="Laplace weight")
    parser.add_argument("--calc_mssim_every", type=int, default=10, help="Calculate MSSIM every n batches")
    parser.add_argument("--calc_starlet_every", type=int, default=15, help="Calculate Starlet every n batches")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads for multihead attention")
    parser.add_argument("--grow_scale", type=float, default=1., help="Grow scale by this factor every 10 epochs")
    parser.add_argument("--double_weights", type=int, default=1, help="Double weights for deep supervision")
    parser.add_argument("--intermediate_attention", type=int, default=1, help="Intermediate attention")
    parser.add_argument("--final_attention", type=int, default=1, help="Final attention")
    parser.add_argument("--learn_diff", type=int, default=1, help="Learn difference")
    parser.add_argument("--psf_min", type=int, default=2, help="PSF min")
    parser.add_argument("--psf_max", type=int, default=8, help="PSF max")
    parser.add_argument("--w_beam", type=float, default=1., help="Beam weight")
    parser.add_argument("--w_flux", type=float, default=0., help="Flux weight")
    parser.add_argument("--dc_blend", type=float, default=0.5, help="DC blend")
    parser.add_argument("--dc_lam", type=float, default=0.1, help="DC lam")
    parser.add_argument("--dc_unroll_steps", type=int, default=0, help="DC unroll steps")
    parser.add_argument("--enforce_nonneg", type=int, default=0, help="Enforce non-negative")
    parser.add_argument("--ssim_weight", type=float, default=0.5, help="Weight of ssim loss")
    parser.add_argument("--recon_weight", type=float, default=0.5, help="Reconstruction weight of original output (~0.5 if scheduling)")
    parser.add_argument("--use_denoise_loss", type=int, default=1, help="Use denoise loss")
    parser.add_argument("--fwd_weight", type=float, default=0.0, help="Fourier forward loss weight")
    parser.add_argument("--x_dc_weight", type=float, default=1., help="X_DC reconstruction loss weight")
    parser.add_argument("--w_spectral", type=float, default=0.0, help="Spectral loss weight (don't do bigger than ~0.05) (~1e-1 if schedule)")
    parser.add_argument("--w_beam_img", type=float, default=0.0, help="Beam image loss weight (~0.5 if schedule)")
    parser.add_argument("--w_low", type=float, default=0.0, help="Low frequency loss weight (~1e-1 if schedule)")
    parser.add_argument("--w_high", type=float, default=0.0, help="High frequency loss weight (~1e-1 if schedule)")
    parser.add_argument("--w_oob", type=float, default=0.0, help="Out-of-band loss weight (~1e-1 if schedule)")
    parser.add_argument("--w_blur", type=float, default=0.1, help="PSF blur loss weight")
    parser.add_argument("--w_starlet", type=float, default=0.25, help="Starlet loss weight")
    parser.add_argument("--w_psd", type=float, default=0., help="High-k PSD loss weight")
    parser.add_argument("--schedule_loss", type=int, default=1, help="Schedule loss")
    parser.add_argument("--ramp_epochs", type=int, default=10, help="Ramp epochs")
    parser.add_argument("--add_film", type=int, default=1, help="Add FiLM layers")
    parser.add_argument("--dc_T", type=int, default=2, help="Unrolled Wiener DC T")
    parser.add_argument("--use_dc", type=int, default=0, help="Use Wiener DC")
    parser.add_argument("--use_white_dc", type=int, default=0, help="Use whitened Wiener DC")
    parser.add_argument("--wiener_init", type=float, default=0.5, help="Wiener init")
    parser.add_argument("--use_self_ensemble", type=int, default=1, help="Use self-ensemble")
    parser.add_argument("--approx_starlet", type=int, default=0, help="Use approximate starlet")
    parser.add_argument("--transform", type=int, default=0, help="Transform data during training")
    parser.add_argument("--use_psf", type=int, default=1, help="Use PSF")
    parser.add_argument("--sum_norm_psf", type=int, default=1, help="Sum PSF = 1")
    
    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=300, help="Maximum training epochs")
    parser.add_argument("--min_epochs", type=int, default=50, help="Minimum training epochs")
    parser.add_argument("--patience", type=int, default=25, help="Patience for early stopping")
    parser.add_argument("--test", type=int, default=0, help="Test mode")
    parser.add_argument("--checkpoint_path", type=str, default="./weights/vireo.ckpt", help="Pretrained model name")

    # wandb project
    parser.add_argument("--wandb_project", type=str, default="vireo", help="W&B project name")

    parser.add_argument("--wandb_entity", type=str, default="chlab", help="W&B Entity")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    # accelerator
    parser.add_argument("--accelerator", type=str, default="mps", help="Accelerator")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main training pipeline for Physics-Informed Neural Network.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    set_seed(args.seed)
    device = get_device(args.accelerator)
    print(f"[INFO] Using device: {device}")
    psf_prefix = "" if bool(args.use_psf) else "no_"
    args.checkpoint_dir = f"./{psf_prefix}{args.checkpoint_dir[2:]}"
    args.final_dir = f"./{psf_prefix}{args.final_dir[2:]}"
    os.makedirs(f"{args.checkpoint_dir}", exist_ok=True)
    os.makedirs(f"{args.final_dir}", exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(args)

    # Logging and callbacks
    print("Setting up WandB")
    run_config = create_wandb_config(args)
    wandb_logger = WandbLogger(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=run_config,
        name=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        log_model=True,
        tags=["disk_clean", "denoising", "continuum"],
    )
    run_config["wandb_name"] = wandb_logger.experiment.name

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.checkpoint_dir}",
            filename="%svireo_{epoch:02d}_{val_mae_main:.4f}_run_%s" % (psf_prefix, wandb_logger.experiment.name),
            save_top_k=1,
            monitor="val/mae_main",
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=f"{args.checkpoint_dir}",
            filename="%svireo_{epoch:02d}_run_%s" % (psf_prefix, wandb_logger.experiment.name),
            save_top_k=1,
            monitor="epoch",
            mode="max",
        ),
        LearningRateMonitor(logging_interval="step"),
        DelayedEarlyStopping(wait_until=args.min_epochs,
            monitor="val/mae_main", min_delta=1e-5, patience=args.patience, verbose=True, mode="min"
        ),
    ]
    
    print("Making trainer")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        min_epochs=min(args.min_epochs, args.max_epochs),
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=device.type,
        log_every_n_steps=1,
        # precision="16-mixed",
        precision="32-true",
        # if device.type == "mps"
        # else "16-mixed",  # MPS works better with 32-bit precision
        deterministic=True,
        # Add these for MPS optimization
        strategy="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print("Compiling model")
    if not bool(args.test):
        model = VIREO(
                lr=args.lr,
                telescope="ska",
                adam_eps=args.adam_eps,
                pct_start=args.pct_start,
                latent_channels=args.latent_channels,
                latent_scale=args.latent_scale,
                ctx_ch=args.ctx_ch,
                ctx_capacity=args.ctx_capacity,
                dropout=args.dropout,
                wandb_name=wandb_logger.experiment.name,
                max_capacity=args.max_capacity,
                recon_loss_type=args.recon_loss_type,
                ssim_weight=args.ssim_weight,
                skip_drop_prob=args.skip_drop_prob,
                skip_dropout_noise=args.skip_dropout_noise,
                add_skip_noise=bool(args.add_skip_noise),
                zero_skips=bool(args.zero_skips),
                schedule_skip_drop=bool(args.schedule_skip_drop),
                max_epochs=args.max_epochs,
                laplace_weight=args.laplace_weight,
                dropblock_prob=args.dropblock_prob,
                dropblock_size=args.dropblock_size,
                window_size=args.window_size,
                scale_out=args.scale_out,
                final_activation=args.final_activation,
                asinh_k=args.asinh_k,
                softclip_a=args.softclip_a,
                calc_mssim_every=args.calc_mssim_every,
                use_inception=bool(args.use_inception),
                num_heads=args.num_heads,
                grow_scale=args.grow_scale,
                which_device=device.type,
                double_weights=bool(args.double_weights),
                intermediate_attention=bool(args.intermediate_attention),
                final_attention=bool(args.final_attention),
                learn_diff=bool(args.learn_diff),
                psf_min=args.psf_min,
                psf_max=args.psf_max,
                n_patches=args.n_patches,
                w_beam=args.w_beam,
                w_flux=args.w_flux,
                dc_blend=args.dc_blend,
                dc_lam=args.dc_lam,
                dc_unroll_steps=args.dc_unroll_steps,
                enforce_nonneg=bool(args.enforce_nonneg),
                recon_weight=args.recon_weight,
                use_denoise_loss=bool(args.use_denoise_loss),
                fwd_weight=args.fwd_weight,
                x_dc_weight=args.x_dc_weight,
                w_spectral=args.w_spectral,
                w_low=args.w_low,
                w_high=args.w_high,
                w_beam_img=args.w_beam_img,
                w_oob=args.w_oob,
                w_blur=args.w_blur,
                w_psd=args.w_psd,
                w_starlet=args.w_starlet,
                schedule_loss=bool(args.schedule_loss),
                ramp_epochs=args.ramp_epochs,
                add_film=bool(args.add_film),
                dc_T=args.dc_T,
                wiener_init=args.wiener_init,
                use_self_ensemble=bool(args.use_self_ensemble),
                calc_starlet_every=args.calc_starlet_every,
                transform=bool(args.transform),
                use_dc=bool(args.use_dc),
                use_white_dc=bool(args.use_white_dc),
                use_psf=bool(args.use_psf),
                sum_norm_psf=bool(args.sum_norm_psf),
        )
    else:
        model = VIREO.load_from_checkpoint(args.checkpoint_path)
        model = model.to(device)
        model.eval()
    
    if bool(args.save_names):
        train_loader.dataset.write_files(f"{args.final_dir}/train_data_{wandb_logger.experiment.name}.txt")
        val_loader.dataset.write_files(f"{args.final_dir}/val_data_{wandb_logger.experiment.name}.txt")
        test_loader.dataset.write_files(f"{args.final_dir}/test_data_{wandb_logger.experiment.name}.txt")
    
    model = model.to(device).train()
    try:
        print("Training")
        torch.use_deterministic_algorithms(True, warn_only=True)

        if not bool(args.test):
            trainer.fit(model, train_loader, val_loader)
        train_passed = True
    except Exception as e:
        wandb.alert(
            title="Training Failed", text=f"Training error: {e}", level=wandb.AlertLevel.ERROR
        )
        print(f"[ERROR] Training interrupted: {e}")
        train_passed = False
        raise
    finally:
        try:
            # Clear memory
            train_loader, val_loader = None, None
            print("Testing")
            trainer.test(model, test_loader)
        except Exception as e:
            wandb.alert(
                title="Testing Failed", text=f"Testing error: {e}", level=wandb.AlertLevel.ERROR
            )
            print(f"[ERROR] Testing interrupted: {e}")
            raise
        finally:
            wandb.finish()
    
    # Save model
    final_model_path = os.path.join(
        f"{args.final_dir}", f"{psf_prefix}vireo_final_model_{wandb_logger.experiment.name}.pt"
    )
    if train_passed:
        torch.save({"model_state_dict": model.state_dict(), "config": run_config}, final_model_path)
        ckpt_path = f"{args.checkpoint_dir}/{psf_prefix}vireo_final_model_{wandb_logger.experiment.name}.ckpt"
        trainer.save_checkpoint(ckpt_path)
    print(f"[INFO] Model saved to {final_model_path}")

if __name__ == "__main__":
    main(parse_arguments())
