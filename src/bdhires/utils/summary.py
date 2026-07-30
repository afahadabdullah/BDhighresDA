"""Startup banner: what is about to be trained, on what, for how long.

Printed once by rank 0 before the first step.  The aim is that a SLURM ``.out``
file is self-documenting six months later -- which store, which statistics file,
which conditioning transform, how many parameters, how many steps -- without
having to reconstruct it from the config and the git history.

Everything here is derived from live objects rather than re-read from the config,
so the banner cannot drift away from what is actually running.
"""

from __future__ import annotations

import subprocess

WIDTH = 78


def _rule(char: str = "=") -> str:
    return char * WIDTH


def _row(label: str, value, indent: int = 2) -> str:
    return f"{' ' * indent}{label:<20}{value}".rstrip()


def _sub(text: str, indent: int = 6) -> str:
    """A continuation line under a labelled row (no label column)."""
    return f"{' ' * indent}{text}".rstrip()


def _si(n: int) -> str:
    """Parameter counts: 38_412_193 -> '38.41 M'."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} B"
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.2f} k"
    return str(n)


def git_commit(short: bool = True) -> str:
    args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


def device_description(device) -> str:
    try:
        import torch

        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            index = device.index if device.index is not None else torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            return (
                f"cuda:{index}  {properties.name}  "
                f"{properties.total_memory / 2**30:.0f} GiB"
            )
    except Exception:  # pragma: no cover
        pass
    return str(device)


def training_summary(
    *,
    cfg: dict,
    config_path: str,
    model,
    train_ds,
    val_ds,
    device,
    world_size: int,
    amp_dtype,
    steps_per_epoch: int,
    total_steps: int,
    stats: dict,
    monitor=None,
    resumed_from: str | None = None,
    start_epoch: int = 0,
) -> str:
    """Build the multi-line startup banner."""
    lines: list[str] = []
    add = lines.append

    add(_rule())
    add(" BDhighresDA - conditional rectified-flow downscaler".ljust(WIDTH))
    add(_rule())

    # -- run ---------------------------------------------------------------
    add(" RUN")
    add(_row("config", config_path))
    add(_row("out_dir", cfg["train"]["out_dir"]))
    add(_row("git commit", git_commit()))
    add(_row("device", device_description(device)))
    add(_row("world size", f"{world_size} process(es)"))
    precision = getattr(amp_dtype, "__str__", lambda: str(amp_dtype))()
    add(_row("precision", f"{precision.replace('torch.', '')} autocast"))
    if resumed_from:
        add(_row("resumed from", f"{resumed_from}  (starting at epoch {start_epoch + 1})"))
    else:
        add(_row("resumed from", "nothing - training from scratch"))

    # -- data --------------------------------------------------------------
    data = cfg["data"]
    add("")
    add(" DATA")
    add(_row("store", data["zarr"]))
    add(_row("stats", data["stats"]))
    precip = stats.get("precip_transform", {})
    add(_row(
        "precip transform",
        f"{precip.get('kind')}  eps={precip.get('eps')}  "
        f"mu={precip.get('mu', 0):.4f}  sd={precip.get('sd', 1):.4f}",
    ))
    cond_kinds = stats.get("cond_transform", {}).get("kinds")
    channel_names = stats.get("cond_channels", [])
    if cond_kinds:
        applied = [
            f"{name}:{kind}"
            for name, kind in zip(channel_names, cond_kinds)
            if kind != "none"
        ]
        add(_row("cond transform", "  ".join(applied) if applied else "identity"))
    else:
        add(_row("cond transform", "identity (pre-v2 stats file)"))
    residual = stats.get("residual", {})
    if residual.get("enabled"):
        summary = stats.get("residual_summary") or {}
        base = summary.get("base_channel_name", "era5_tp")
        add(_row(
            "target",
            f"RESIDUAL: T(CHIRPS) - T({base}), standardised "
            f"(mean {residual.get('mean', 0):.4f}, std {residual.get('std', 1):.4f})",
        ))
        if summary:
            add(_sub(
                f"encoded target mean {summary.get('encoded_mean', float('nan')):+.4f}  "
                f"std {summary.get('encoded_std', float('nan')):.4f}  "
                f"|max| {summary.get('encoded_abs_max', float('nan')):.1f}"
            ))
            add(_sub(
                f"base explains r={summary.get('base_correlation', float('nan')):.3f} "
                f"of the transformed target"
            ))
        add(_sub("zero residual reproduces ERA5, so the model has a skill floor"))
    else:
        add(_row("target", "ABSOLUTE: T(CHIRPS)"))
    years = data["years"]
    add(_row(
        "train",
        f"{years['train'][0]}-{years['train'][1]}   {len(train_ds):,} days",
    ))
    add(_row(
        "val",
        f"{years['val'][0]}-{years['val'][1]}   {len(val_ds):,} days",
    ))
    add(_row(
        "crop",
        f"{data['crop']}x{data['crop']} random from {train_ds.H}x{train_ds.W}"
        f"   min land {data.get('min_valid_fraction', 0.0):.0%}",
    ))
    n_static = train_ds.static.shape[0]
    n_season = train_ds.total_cond_channels - train_ds.n_cond - n_static
    add(_row("conditioning", f"{train_ds.total_cond_channels} channels"))
    add(_sub(f"{train_ds.n_cond:>3d} ERA5    "
             f"{' '.join(channel_names) or 'unnamed'}"))
    static_names = stats.get("static_channels", [])
    add(_sub(f"{n_static:>3d} static  "
             f"{' '.join(static_names) or 'unnamed'}"))
    if n_season:
        add(_sub(f"{n_season:>3d} season  sin_doy cos_doy"))

    # -- model -------------------------------------------------------------
    add("")
    add(" MODEL")
    add(_row(
        "UNet",
        f"base={model.base_channels_arg}  mult={list(model.channel_mult)}  "
        f"blocks={model.num_res_blocks}  heads={model.num_heads}  "
        f"dropout={model.dropout}",
    ))
    add(_sub("level   resolution   channels   attn"))
    for level in model.levels():
        name = str(level["level"])
        add(_sub(
            f"{name:<7s} {level['resolution']:>4d}x{level['resolution']:<5d} "
            f"{level['channels']:>8d}   {'yes' if level['attention'] else '-'}"
        ))
    add(_row("parameters", "by top-level module"))
    for name, count in model.parameters_by_module().items():
        add(_sub(f"{name:<14s} {_si(count):>10s}"))
    total = model.num_parameters
    add(_sub(f"{'TOTAL':<14s} {_si(total):>10s}   "
             f"({_si(model.num_trainable_parameters)} trainable)"))
    add(_row(
        "weights",
        f"{total * 4 / 2**20:.0f} MiB fp32   "
        f"+{total * 4 / 2**20:.0f} MiB EMA shadow   "
        f"+{total * 8 / 2**20:.0f} MiB AdamW state",
    ))

    # -- optimisation ------------------------------------------------------
    train = cfg["train"]
    add("")
    add(" OPTIMISATION")
    add(_row(
        "AdamW",
        f"lr={train['lr']:.2e}  wd={train['weight_decay']}  "
        f"betas=(0.9, 0.999)  clip={train['grad_clip']}",
    ))
    add(_row(
        "schedule",
        f"{train['warmup_steps']:,} warmup steps then cosine to 0",
    ))
    add(_row(
        "batch",
        f"{train['batch_size']} per process"
        + (f"  ({train['batch_size'] * world_size} global)" if world_size > 1 else ""),
    ))
    add(_row(
        "budget",
        f"{train['epochs']} epochs x {steps_per_epoch:,} steps/epoch = "
        f"{total_steps:,} steps",
    ))
    horizon = int(1 / (1 - train["ema_decay"])) if train["ema_decay"] < 1 else 0
    add(_row("EMA", f"decay={train['ema_decay']}  (~{horizon:,} step horizon)"))
    add(_row(
        "cond_dropout",
        f"{train['cond_dropout']}  -> buys the unconditional branch used by "
        f"CFG at sampling",
    ))
    add(_row(
        "loss",
        "masked flow-matching MSE"
        + ("  (logit-normal t)" if train.get("logit_normal_t", True) else "  (uniform t)"),
    ))
    add(_row("checkpoints", f"every {train['ckpt_every']} epochs -> best.pt, last.pt"))

    # -- validation --------------------------------------------------------
    add("")
    add(" VALIDATION")
    if monitor is None:
        add(_row("sampled", "disabled - selecting on flow-matching loss only"))
        add(_sub("the FM loss is dominated by its own sampling noise;"))
        add(_sub("enable validation.* to select on CRPS instead"))
    else:
        mcfg = monitor.cfg
        fires = sum(
            1
            for e in range(train["epochs"])
            if (e + 1) >= mcfg.start_epoch and (e + 1) % mcfg.every == 0
        )
        add(_row(
            "sampled",
            f"every {mcfg.every} epochs from epoch {mcfg.start_epoch} "
            f"({fires} evaluations)",
        ))
        for case in monitor.cases:
            add(_sub(
                f"{case.date}  q{int(round(case.quantile * 100)):02d}  "
                f"domain mean {case.domain_mean_mm:.2f} mm/day"
            ))
        nfe = (
            mcfg.members * mcfg.n_steps * 2
            * (2 if mcfg.cfg_scale != 1.0 else 1)
            * max(1, len(monitor.cases))
        )
        add(_row(
            "sampler",
            f"{mcfg.members} members  {mcfg.n_steps} steps  "
            f"CFG w={mcfg.cfg_scale:g}  -> ~{nfe:,} NFE per evaluation",
        ))
        add(_row("selection", "best.pt chosen on sampled CRPS"))
        add(_row("outputs", f"{cfg['train']['out_dir']}/validation/"))

    add(_rule())
    return "\n".join(lines)
