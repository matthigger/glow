"""Centralized AWS pricing constants and conversion helpers."""

COST_PER_VCPU_HOUR = 0.02   # us-east-1 Fargate Spot, reviewed 2026-04
SPINUP_SECONDS = 180         # measured from Batch describe_jobs timestamps


def vcpu_hours_to_dollars(hours):
    """Convert vCPU-hours to USD at the configured rate."""
    return hours * COST_PER_VCPU_HOUR


def wall_to_vcpu_hours(wall_sec, vcpus, include_spinup=True):
    """Convert wall-clock seconds to vCPU-hours.

    Args:
        wall_sec: wall-clock compute time per job (seconds)
        vcpus: vCPUs per job
        include_spinup: if True, add SPINUP_SECONDS to each job's wall time
            to reflect total cost (spinup is billed).
    """
    total_sec = wall_sec + (SPINUP_SECONDS if include_spinup else 0)
    return total_sec * vcpus / 3600.0


def format_cost_estimate(n_jobs, perm_sec, perms_per_job, vcpus):
    """Format a one-line cost estimate string.

    Returns a string like:
        "$X.XX (Y.Y vCPU-hrs, N jobs x M min spinup)"
    """
    compute_sec_per_job = perm_sec * perms_per_job
    vcpu_hrs = n_jobs * wall_to_vcpu_hours(compute_sec_per_job, vcpus)
    cost = vcpu_hours_to_dollars(vcpu_hrs)
    return (f"${cost:.2f} ({vcpu_hrs:.1f} vCPU-hrs, "
            f"{n_jobs} jobs x {SPINUP_SECONDS // 60} min spinup)")
