"""AWSConfig: AWS Batch + S3 settings passed to driver_aws + infra CLI.

JSON serialisation lives here (``to_file`` / ``from_file``) so the
project root can hold a ``.glow_aws_config`` and the CLI doesn't need
to know how to construct one.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List


DEFAULT_CONFIG_PATH = '.glow_aws_config'


@dataclass
class AWSConfig:
    s3_bucket: str
    s3_prefix: str
    job_queue: str
    job_definition: str
    region: str = 'us-east-1'
    vcpus: int = 1
    memory_mb_tiers: List[int] = field(
        default_factory=lambda: [2000, 4000, 8000])
    max_concurrent: int = 4000
    timeout_minutes: int = 60
    retry_attempts: int = 3
    poll_seconds: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    def to_file(self, path=DEFAULT_CONFIG_PATH) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')

    @classmethod
    def from_dict(cls, d: dict) -> 'AWSConfig':
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f'unknown AWSConfig keys: {sorted(unknown)}')
        return cls(**d)

    @classmethod
    def from_file(cls, path=DEFAULT_CONFIG_PATH) -> 'AWSConfig':
        return cls.from_dict(json.loads(Path(path).read_text()))
