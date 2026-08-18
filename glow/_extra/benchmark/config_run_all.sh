#!/usr/bin/env bash
# Run the whole CONFIG catalogue, in lanes that do not contend for a resource.
#
# Usage:
#     glow/_extra/benchmark/config_run_all.sh
#     PYTHON=/path/to/python glow/_extra/benchmark/config_run_all.sh
#
# Resumable: the driver runs only the cells its records do not already hold,
# so a re-run after a failure (or an interrupt) continues rather than repeats.
#
# WHY LANES. A device leaf -- any recipe carrying config.GLOW_FIT_PARAMS,
# gpu='auto' -- must run at driver n_jobs=1, since driver.check_fit_params
# refuses to put two fits on one CUDA context. So GLOW's caches run serial in
# the driver and take their parallelism from GLOW_FIT_PARAMS instead. The
# voxel-wise arms never touch the device, so they take the driver's own -j and
# run beside GLOW rather than after it. At paper scale a cell's GLOW fit and
# its three voxel-wise fits cost about the same wall time, so overlapping the
# two lanes takes the voxel-wise arms off the critical path entirely.
#
# THE LANES.
#   1  the voxel-wise arms over the six shared-grid caches, parallel across
#      data cells.
#   2  GLOW over those same six, then prune, then sweep_llr_glow_tune, then
#      sweep_n_perm_inner -- each an independent step, so a later one still
#      runs if an earlier fails. Serial in the driver, feeding the device.
#   3  the caches with no per-method axis and no device, at full parallelism.
#   4  every timing cache, last and alone. The 1perm leaves pin themselves to
#      one core, but sharing the box with lane 3 would contend for cache and
#      memory bandwidth and skew a growth rate, so they get an undisturbed
#      machine at the cost of some wall time.
#
# Lanes 1 and 2 run concurrently; 3 and 4 run after, and apart from each
# other. Every step goes through run_step, which records a failure and returns
# 0 regardless -- one cache crashing (a non-convergent effect plant, an OOM at
# full-brain num_vox) costs that step, not the remaining lanes. The script's
# own exit status is nonzero iff anything failed, checked once at the end.
#
# No -e: run_step is what handles a failing step, and -e would abandon the
# remaining lanes on the first one.
set -uo pipefail
cd "$(dirname "$0")/../../.."

# an interpreter with glow importable. The venv is this project's usual one
# (the system python lacks nibabel / dipy); PYTHON overrides it anywhere else.
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$HOME/venv_glow/bin/python" ]; then
        PYTHON="$HOME/venv_glow/bin/python"
    else
        PYTHON=python3
    fi
fi

BENCH=("$PYTHON" -m glow._extra.benchmark)

# Cores held back for lane 2. Its driver runs serial, but the fit underneath
# takes config.GLOW_FIT_N_JOBS workers and a device kept fed needs about that
# many threads, so lane 1 must not claim them. Keep in step with
# config.GLOW_FIT_N_JOBS.
GLOW_CORES=10
LANE1_JOBS=$(( $(nproc) - GLOW_CORES ))
[ "$LANE1_JOBS" -lt 1 ] && LANE1_JOBS=1

# the caches sharing config.RUN_ANA_LIST: one GLOW variant plus the
# voxel-wise arms, so they are the caches with a per-method axis to split on.
SHARED_GRID_CACHES=(null sweep_llr sweep_extent sweep_b sweep_nimg smoke)

# the reported GLOW variant (config.REPORTED_GLOW_LABEL) -- the only GLOW
# entry on that shared grid.
GLOW_METHOD=GLOW-Focus-greedy

VOXEL_METHODS=(--method VBA --method VBA-TFCE --method CET)

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

run_step() {
    local label=$1; shift
    echo "=== $label: $* ==="
    if "$@"; then
        echo "$label" >> "$LOG_DIR/ok"
    else
        echo "$label (exit $?)" >> "$LOG_DIR/failed"
        echo "!!! $label failed, continuing" >&2
    fi
}

# Lane 1: the voxel-wise arms, parallel across data cells.
run_step lane1 "${BENCH[@]}" "${VOXEL_METHODS[@]}" -j "$LANE1_JOBS" \
    "${SHARED_GRID_CACHES[@]}" &
lane1=$!

# Lane 2: GLOW, serial in the driver, its workers and the device underneath.
(
    run_step lane2a "${BENCH[@]}" --method "$GLOW_METHOD" -j 1 \
        "${SHARED_GRID_CACHES[@]}"
    run_step lane2b "${BENCH[@]}" -j 1 prune
    run_step lane2c "${BENCH[@]}" -j 1 sweep_llr_glow_tune
    # the inner-draw sweep parallelises its own walk like every other GLOW
    # leaf, but its capture is what carries the cost, so --no-gpu -j2 finishes
    # it sooner on a box with cores to spare (the device fit is serial here).
    run_step lane2d "${BENCH[@]}" -j 1 sweep_n_perm_inner
) &
lane2=$!

wait "$lane1" "$lane2"

# Lane 3: no per-method axis, no device -- full parallelism.
run_step lane3 "${BENCH[@]}" -j -1 segment segment_perc_llr vba_stat

# Lane 4: every timing cache, on an undisturbed machine.
run_step lane4 "${BENCH[@]}" runtime_num_vox 'runtime_1perm_*'

echo
if [ -f "$LOG_DIR/ok" ]; then
    echo "OK:"; cat "$LOG_DIR/ok"
fi
status=0
if [ -f "$LOG_DIR/failed" ]; then
    echo "FAILED:"; cat "$LOG_DIR/failed"
    status=1
fi
exit "$status"
