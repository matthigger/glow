"""AWSConfig: AWS Batch + S3 settings passed to driver_aws + infra CLI.

JSON serialisation lives here (to_file / from_file) so the project root
can hold a .glow_aws_config and the CLI doesn't need to know how to
construct one.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List


DEFAULT_CONFIG_PATH = '.glow_aws_config'


def s3_key(prefix: str, *parts: str) -> str:
    """Join an ``s3_prefix`` with key parts, tolerating an empty prefix.

    With an empty ``s3_prefix`` objects land at the bucket root rather than
    under a leading-slash key like ``/jobs/...``.
    """
    return '/'.join(p for p in (prefix, *parts) if p)


@dataclass
class AWSConfig:
    """AWS Batch + S3 settings shared by driver_aws and the infra CLI.

    Most fields are self-documenting via their annotations.  The less
    obvious ones:

    Attributes:
        s3_prefix (str): key prefix under s3_bucket for all glow objects.
        memory_mb_tiers (List[int]): per-attempt memory limits; the
            driver re-runs OOM-killed trials at the next larger tier.
        max_concurrent (int): compute-environment maxvCpus ceiling.
        timeout_minutes (int): per-attempt wall-clock limit.
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
    timeout_minutes: int = 60
    retry_attempts: int = 3
    poll_seconds: int = 10

    def to_dict(self) -> dict:
        """Plain-dict view of this config (dataclass asdict)."""
        return asdict(self)

    def to_file(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        """Write this config to path as indented JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')

    @classmethod
    def from_dict(cls, d: dict) -> 'AWSConfig':
        """Build an AWSConfig from a dict, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f'unknown AWSConfig keys: {sorted(unknown)}')
        return cls(**d)

    @classmethod
    def from_file(cls, path: str = DEFAULT_CONFIG_PATH) -> 'AWSConfig':
        """Build an AWSConfig from a JSON file at path."""
        return cls.from_dict(json.loads(Path(path).read_text()))
