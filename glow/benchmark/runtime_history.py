"""Post-hoc runtime log: append one JSON record per completed cloud job.

The idea is to accumulate (predicted_sec, actual_sec) pairs in the wild so
future sessions can refit the Lasso or recalibrate the 99.99% safety factor
against real production data instead of the WGN training grid.

Records land in ``~/.local/share/glow/results/runtime_history/*.json`` —
write-only for now; no refit/analysis logic lives here.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from platformdirs import user_data_dir


def _history_dir():
    d = Path(user_data_dir('glow', 'glow_author')) / 'results' / 'runtime_history'
    d.mkdir(parents=True, exist_ok=True)
    return d


def record(*, config_label, analysis_type, num_vox, b, num_img, n_perm,
           predicted_sec, actual_sec, job_id=None):
    """Append one runtime record. Safe to call with None fields."""
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config_label': config_label,
        'analysis_type': analysis_type,
        'num_vox': int(num_vox) if num_vox is not None else None,
        'b': int(b) if b is not None else None,
        'num_img': int(num_img) if num_img is not None else None,
        'n_perm': int(n_perm) if n_perm is not None else None,
        'predicted_sec': float(predicted_sec) if predicted_sec is not None else None,
        'actual_sec': float(actual_sec) if actual_sec is not None else None,
        'job_id': job_id,
    }

    ts = entry['timestamp'].replace(':', '').replace('.', '')
    fname = f'{ts}_{config_label or "unknown"}.json'
    path = _history_dir() / fname
    with open(path, 'w') as f:
        json.dump(entry, f, indent=2)
    return path
