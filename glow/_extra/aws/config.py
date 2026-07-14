"""AWSConfig: AWS Batch + S3 settings shared by the driver, worker, infra CLI.

JSON serialisation lives here (to_file / from_file). The default path is the
per-user config directory (platformdirs user_config_dir, e.g.
~/.config/glow/aws_config.json on Linux) -- the config peer of the
user_data_dir glow already writes benchmark cache / records / results to.
Callers override it with the --config CLI flag or the path argument.

The worker reads the same bucket / prefix / region to sync the shared records
folder to and from S3 (see glow._extra.aws.s3); the driver reads the Batch
queue / definition / memory tiers to submit and escalate array jobs (see
glow._extra.aws.driver).
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List

from platformdirs import user_config_dir


# per-user config location (XDG ~/.config/glow on Linux), the config peer of
# the user_data_dir glow already uses for the cache / records / results.
DEFAULT_CONFIG_PATH = str(Path(user_config_dir('glow', 'glow_author')) /
                          'aws_config.json')


def s3_key(prefix: str, *parts: str) -> str:
    """Join an s3_prefix with key parts, tolerating an empty prefix.

    With an empty s3_prefix objects land at the bucket root rather than under
    a leading-slash key like /records/....
    """
    return '/'.join(p for p in (prefix, *parts) if p)


@dataclass
class AWSConfig:
    """AWS Batch + S3 settings shared by the driver, worker, and infra CLI.

    Most fields are self-documenting via their annotations. The less obvious
    ones:

    Attributes:
        s3_prefix (str): key prefix under s3_bucket for all glow objects (the
            shared records and per-run manifests live beneath it).
        memory_mb_tiers (List[int]): per-attempt memory limits; the driver
            re-runs OOM-killed cells at the next larger tier. The first tier
            matches 2 GB/vCPU compute-optimized nodes so a 1-vcpu job packs
            onto a single vCPU.
        max_concurrent (int): compute-environment maxvCpus ceiling.
        timeout_minutes (int): per-attempt wall-clock limit. A single data
            cell runs its whole effect x analysis subtree serially (the
            heaviest, sweep_extent, is hours), and a Spot-killed attempt
            resumes from the synced cache on retry (see the driver / s3
            modules), so a generous ceiling is safe.
        retry_attempts (int): per-job Batch attempt budget. A Spot reclaim is
            transient, not a real failure, so the budget mostly buffers host
            loss (the driver's evaluateOnExit retries it in place, warm-
            resuming from cache); set high enough that a short correlated
            capacity crunch -- several reclaims in a row -- does not exhaust it.
            OOM is held separate (it EXITs and the driver escalates memory), so
            a larger budget never re-loops an OOM.
        poll_seconds (int): interval between describe_jobs status polls.
    """

    s3_bucket: str
    job_queue: str
    job_definition: str
    s3_prefix: str = ''
    region: str = 'us-east-1'
    vcpus: int = 1
    memory_mb_tiers: List[int] = field(
        default_factory=lambda: [2000, 4000, 8000])
    max_concurrent: int = 4000
    timeout_minutes: int = 720
    retry_attempts: int = 6
    poll_seconds: int = 10

    def to_dict(self) -> dict:
        """Plain-dict view of this config (dataclass asdict)."""
        return asdict(self)

    def to_file(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        """Write this config to path as indented JSON, creating parent dirs."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + '\n')

    @classmethod
    def from_dict(cls, d: dict) -> 'AWSConfig':
        """Build an AWSConfig from a dict, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f'unknown AWSConfig keys: {sorted(unknown)}')
        return cls(**d)

    @classmethod
    def from_file(cls, path: str = DEFAULT_CONFIG_PATH) -> 'AWSConfig':
        """Build an AWSConfig from a JSON file at path."""
        return cls.from_dict(json.loads(Path(path).read_text()))
