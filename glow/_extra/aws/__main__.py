"""CLI for AWS provisioning: ``python -m glow._extra.aws <subcommand>``.

A short alias for the provisioning CLI (glow._extra.aws.infra): bootstrap /
setup / teardown / status / clear_storage / clear_jobs / pause / resume -- the
commands that manage the AWS resources (IAM, ECR, the Batch compute-env / queue
/ job-definition, S3).

Running a benchmark sweep -- locally or on AWS -- is the benchmark CLI's job,
not this one: ``python -m glow._extra.benchmark --aws <names>`` (see
glow._extra.benchmark). Keeping the run path in one place means there is a
single CLI for a benchmark run, and this one is only for the (separate, rare)
provisioning lifecycle.
"""
from . import infra

if __name__ == '__main__':
    infra.main()
