"""NumPy-only paired deterministic uncertainty for the model paper."""
import numpy as np

FINAL = "dense_s6_bwdb_r4"


def paired_intervals(data, predictions, block_days, resamples, seed):
    """Bootstrap pooled RMSE/MAE gains by resampling whole day blocks.

    All stations on a day move together. Recompute nonlinear RMSE in each
    resample. Station-day weighting matches the headline deterministic table.
    Bonferroni intervals cover the six primary product/metric contrasts for
    each block-width sensitivity analysis. No model-selection uncertainty.
    """
    dates = np.unique(data["date"])
    split = np.where(np.diff(dates).astype("timedelta64[D]").astype(int) > 1)[0] + 1
    segments = np.split(np.arange(len(dates)), split)
    day_index = np.searchsorted(dates, data["date"])
    n = np.bincount(day_index).astype(float)
    error = {s: p - data["truth"] for s,p in predictions.items()}
    sums = {s: (np.bincount(day_index, weights=e**2), np.bincount(day_index, weights=np.abs(e))) for s,e in error.items()}
    output = []
    for reference in [s for s in predictions if s != FINAL]:
        for width in block_days:
            rng = np.random.default_rng(seed)
            gains = np.empty((resamples,2))
            # Bounded memory: totals for at most 256 resamples at once.
            for start in range(0, resamples, 256):
                batch = min(256, resamples-start)
                totals = np.zeros((batch,5))
                for segment in segments:
                    length = len(segment); size = min(width,length)
                    starts = rng.integers(0,length,(batch,int(np.ceil(length/size))))
                    index = segment[((starts[...,None]+np.arange(size)).reshape(batch,-1)[:,:length]) % length]
                    values = [n, sums[reference][0], sums[FINAL][0], sums[reference][1], sums[FINAL][1]]
                    for col,v in enumerate(values):
                        totals[:,col] += v[index].sum(axis=1)
                gains[start:start+batch,0] = np.sqrt(totals[:,1]/totals[:,0]) - np.sqrt(totals[:,2]/totals[:,0])
                gains[start:start+batch,1] = (totals[:,3]-totals[:,4])/totals[:,0]
            for col, metric in enumerate(("rmse", "mae")):
                low, high = np.percentile(gains[:,col],[2.5,97.5])
                family = 6 if reference in ("chirps","imerg","cpc") else 0
                adj = np.percentile(gains[:,col], [100*.05/(2*family),100*(1-.05/(2*family))]) if family else [None,None]
                point = (np.sqrt(sums[reference][0].sum()/n.sum())-np.sqrt(sums[FINAL][0].sum()/n.sum())
                         if col == 0 else (sums[reference][1].sum()-sums[FINAL][1].sum())/n.sum())
                output.append({"reference": reference, "candidate": FINAL, "metric": metric,
                               "gain_mm_day": float(point), "ci_low": float(low), "ci_high": float(high),
                               "family_size": family, "family_ci_low": adj[0], "family_ci_high": adj[1],
                               "block_days": width, "n_resamples": resamples, "n_days": len(dates),
                               "n_segments": len(segments), "n": int(n.sum()), "seed": seed,
                               "weighting": "pooled station-days; all sites resampled together within day"})
    return output



def checkpoint_metadata(path, expected_sha256, loader=None):
    """Read scalar checkpoint metadata only after checking its pinned identity.

    This does not instantiate a network or infer the checkpoint epoch from a
    validation-curve minimum. Restricted Torch loading never falls back to
    unrestricted pickle. An injected loader supports NumPy-only unit tests.
    """
    import hashlib
    from pathlib import Path
    path = Path(path)
    if not path.is_file():
        return {"status": "checkpoint_missing", "path": str(path)}
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    identity = digest.hexdigest()
    if identity != expected_sha256:
        raise ValueError('checkpoint differs from the evaluated paper SHA-256')
    if loader is None:
        try:
            import torch
        except ImportError:
            return {"status": "metadata_reader_unavailable", "sha256": identity,
                    "note": "Run on the existing Torch environment; epoch is not inferred from history."}
        loader = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    try:
        payload = loader(path)
    except Exception as error:
        raise ValueError('restricted checkpoint metadata read failed: '+str(error)) from error
    if not isinstance(payload, dict):
        raise ValueError('checkpoint must contain a metadata mapping')
    epoch = payload.get('epoch')
    if epoch is None:
        return {"status": "epoch_unrecorded", "sha256": identity}
    if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
        raise ValueError('checkpoint epoch must be a nonnegative zero-based integer')
    result = {"status": "verified_checkpoint_epoch", "path": str(path.resolve()),
              "sha256": identity, "epoch_zero_based": int(epoch),
              "completed_epoch": int(epoch)+1,
              "selected_by": payload.get('selected_by'), "weights": payload.get('weights')}
    for name in ('step', 'val_loss', 'best_val_loss', 'crps', 'best_crps'):
        value = payload.get(name)
        if value is None or isinstance(value, (str, int, float, bool)):
            result[name] = None if isinstance(value, float) and not np.isfinite(value) else value
    result['note'] = 'Epoch is read from the content-verified checkpoint, not the plotted validation minimum.'
    return result
