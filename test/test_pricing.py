"""Unit tests for glow.aws.pricing helpers."""

from pathlib import Path

import pytest

from glow.aws.pricing import (
    COST_PER_VCPU_HOUR,
    SPINUP_SECONDS,
    format_cost_estimate,
    vcpu_hours_to_dollars,
    wall_to_vcpu_hours,
)


def test_vcpu_hours_to_dollars_unit_rate():
    assert vcpu_hours_to_dollars(1.0) == pytest.approx(COST_PER_VCPU_HOUR)
    assert vcpu_hours_to_dollars(1.0) == pytest.approx(0.02)


def test_vcpu_hours_to_dollars_scales_linearly():
    assert vcpu_hours_to_dollars(10.0) == pytest.approx(10.0 * COST_PER_VCPU_HOUR)
    assert vcpu_hours_to_dollars(0.0) == pytest.approx(0.0)


def test_wall_to_vcpu_hours_no_spinup():
    # 3600 seconds * 1 vcpu == 1 vcpu-hour
    assert wall_to_vcpu_hours(3600, 1, include_spinup=False) == pytest.approx(1.0)


def test_wall_to_vcpu_hours_with_spinup():
    # 3420 + 180 spinup = 3600 sec -> 1 vcpu-hour
    assert SPINUP_SECONDS == 180
    assert wall_to_vcpu_hours(3420, 1, include_spinup=True) == pytest.approx(1.0)


def test_wall_to_vcpu_hours_multi_vcpu():
    # 1800 sec * 4 vcpus == 2 vcpu-hours (no spinup)
    assert wall_to_vcpu_hours(1800, 4, include_spinup=False) == pytest.approx(2.0)


def test_format_cost_estimate_contents():
    # 10 jobs, each 60s/perm * 10 perms = 600s compute
    # wall (with 180s spinup) per job = 780s
    # total vcpu-hrs = 10 * 780 * 2 / 3600 = 4.333...
    # cost = 4.333 * 0.02 = $0.087
    s = format_cost_estimate(n_jobs=10, perm_sec=60, perms_per_job=10, vcpus=2)
    assert isinstance(s, str)
    assert s.startswith('$')
    assert 'vCPU-hrs' in s
    assert '10 jobs' in s
    assert f'{SPINUP_SECONDS // 60} min spinup' in s


def test_format_cost_estimate_numeric_correctness():
    # single-job, 3420s compute, 1 vcpu: vcpu-hrs = (3420+180)*1/3600 = 1
    # cost = $0.02
    s = format_cost_estimate(n_jobs=1, perm_sec=3420, perms_per_job=1, vcpus=1)
    assert '$0.02' in s
    assert '1.0 vCPU-hrs' in s


def test_no_hardcoded_002_outside_pricing():
    """Regression guard: the literal ``0.02`` should not appear in glow/ .py
    files outside ``pricing.py`` as a cost constant.

    Specific non-cost uses that coincidentally equal 0.02 are whitelisted.
    """
    glow_dir = Path(__file__).resolve().parent.parent / 'glow'
    assert glow_dir.is_dir(), glow_dir

    # (relative path, substring identifying the allowed line)
    whitelist = {
        # axes-fraction annotation coordinates in runtime_plot spinup label
        ('benchmark/runtime_plot.py', "xy=(0.98, 0.02)"),
        # plot y-axis limits for p-value plots
        ('benchmark/plot_mancova_vba.py', '-0.02, 1.02'),
        # viewer scatter annotation offset
        ('viewer/scatter.py', 'x=0.02'),
    }

    offenders = []
    for path in glow_dir.rglob('*.py'):
        rel = path.relative_to(glow_dir).as_posix()
        if rel == 'aws/pricing.py':
            continue
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                if '0.02' not in line:
                    continue
                if any(rel == wl_rel and wl_sub in line
                       for wl_rel, wl_sub in whitelist):
                    continue
                offenders.append(f'{rel}:{lineno}: {line.rstrip()}')

    assert not offenders, (
        'Hardcoded 0.02 found outside glow/aws/pricing.py:\n'
        + '\n'.join(offenders)
        + '\nUse COST_PER_VCPU_HOUR from glow.aws.pricing instead.'
    )


