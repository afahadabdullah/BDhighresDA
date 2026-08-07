#!/usr/bin/env python
"""SUPERSEDED -- do not use. See configs/train_h100_cpc_v3_hurdle_cfg2.yaml.

This was going to reconstruct the validation monitor standalone in order to
resample one checkpoint at several guidance weights. It was abandoned before
being run: the monitor needs the full dataset, transform, residual spec and
climatology plumbing that ``scripts/train.py:build_monitor`` already assembles,
and duplicating that here would have meant shipping a second, untested copy of
setup code whose only job is to be identical to the first.

The same question is answered by resuming the finished checkpoint through the
ordinary training entry point with ``validation.cfg_scale`` changed, which
exercises only code paths that already run every epoch:

    mkdir -p runs/prior_h100_cpc_v3_hurdle_cfg2
    cp runs/prior_h100_cpc_v3_hurdle/last.pt runs/prior_h100_cpc_v3_hurdle_cfg2/
    bash slurm/submit_train_cpc_v3_hurdle_cfg2_gh200.sh

Compare the resulting spread and wet fraction against the w=1 values that run
reported at epoch 149 (spread 23.35, wet 0.994, error +0.298, corr 0.286).
"""

import sys

sys.exit(__doc__)
