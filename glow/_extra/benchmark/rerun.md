# Reproducing the glow benchmarks

Everything in the paper comes from one catalogue of declared experiments
(`glow._extra.benchmark.config.CONFIG`) driven by one CLI. This file is the
recipe. Every path and command below is relative to the repository root.

Two inputs, one output. The inputs are this repository and the imaging data.
The output is a tree of provenance records -- one JSON per computed leaf,
recording that leaf's inputs, score and timing -- and every figure and table
in the paper is drawn from it.

| | artifact | where |
|---|---|---|
| input | this repository | [github.com/matthigger/glow](https://github.com/matthigger/glow) at commit `2f0c04e` |
| input | HCP-YA imaging data, 100 subjects x 6 maps | [Zenodo 10.5281/zenodo.20736221](https://doi.org/10.5281/zenodo.20736221) |
| output | provenance records, version 2 | [Zenodo 10.5281/zenodo.22664285](https://doi.org/10.5281/zenodo.22664285) |

Version 2 is the corpus the paper reports. Version 1 holds an earlier corpus,
on a cache layer this code no longer reads, with its plants drawn from
different seeds.

### The corpus

| | |
|---|---|
| commit | `2f0c04e` on `main` |
| seeds | 50 (`null` 1000; `vba_tune`, `oracle_stat` 10) |
| tuned | 1 - Wilks at 2 mm for VBA, VBA-TFCE, CET; z for VBA-TFCE only |
| holes | 22 HCP b=1 cells (seeds 12, 15) fail the plant check (`cell.LLR_RTOL`) and are skipped; see [Two seeds that missed their plant](#two-seeds-that-missed-their-plant) |
| not in the paper | `sweep_llr_wgn_sphere`, `sweep_extent`, `sweep_b`, `sweep_n_perm_inner`, `oracle_stat`, `smoke`, `runtime_1perm_n_perm_fwer` |

A record does not name the commit that computed it. The records are
consistent with `2f0c04e`: there, every paper cache's declared cells resolve
to recorded leaves bar the holes, recomputed leaves reproduce their records
bit for bit (16,439 of the 16,444 we recomputed; see [How we checked
ours](#how-we-checked-ours)), and `plot` redraws every paper figure and
table from the records alone, bar one table whose generator is not on
`main`. Later commits on `main` change no recorded number unless this file
says so.

To check a tree against the store, count each cache's incomplete cells under
that tree, with no seed cap:

```bash
python -c "
from glow._extra.benchmark import results, store
store.RECORDER.load()
for n in ['sweep_llr', 'sweep_extent', 'sweep_b', 'null', 'segment', 'prune',
          'sweep_n_perm_inner', 'vba_stat', 'vba_tune']:
    print(n, len(results.incomplete_cell_indices(n)))"
```

A consistent tree reports only the holes: sweep_llr 11, sweep_extent 12,
sweep_b 1, segment 11, prune 11, sweep_n_perm_inner 3, and 0 for null,
vba_stat and vba_tune, 22 distinct cells in all. That check sees every
change to a declared recipe or cell, not a change to code running underneath
one; recomputing leaves (`validate_records`) is what sees that.

The tuning caches ran first, and that ordering was load-bearing rather than a
preference: `config.SMOOTH_FWHM_BEST` is read off `vba_tune`, and wiring a
kernel width into an arm changes that arm's repr, which is its recipe
identity, so every leaf it had already recorded would re-key. Tune before the
reported caches run, not after.

`vba_tune` chooses the statistic, the z-scoring and the width together, since
choosing one at a time is circular (each sweep would assume the other's
answer). All three arms came out on 1 - Wilks at 2 mm. rank(H) = 1 leaves
VBA's and CET's rejection sets invariant to the statistic, so they tie across
the pool and adopt the choice of VBA-TFCE, the one arm that can tell the
statistics apart. VBA-TFCE's own maximum ties 1 - Wilks with Pillai's trace,
which at rank(H) = 1 are the same function of the lone eigenvalue; it takes
Wilks by the fixed statistic order.

You do not have to generate the output to read it. Ours is published, so the
figures redraw from it with no compute and without the imaging data -- see
[Using our records](#using-our-records), or [One CSV per
figure](#one-csv-per-figure) if you only want the plotted numbers. Generating
your own is what the rest of this file is for; it is worth reading [why it is
not bitwise reproducible](#why-it-is-not-bitwise-reproducible) first.

## Install

```bash
git clone https://github.com/matthigger/glow && cd glow
python3.12 -m venv ~/venv_glow
~/venv_glow/bin/pip install -r requirements-lock.txt
~/venv_glow/bin/pip install -e . --no-deps
```

`requirements-lock.txt` pins every package in the environment that produced
the records, for CPython 3.12; where the OS ships no 3.12, make one with conda
or uv and build the venv from it. Optional: **FSL** (the TFCE arm uses
`fslmaths` where it is installed) and an NVIDIA card (the GLOW arm uses it
where it is visible).

## Data

```bash
python -c "from glow._extra.benchmark import hcp; hcp.ensure_hcp_data()"
```

Prints the WU-Minn HCP Open Access Data Use Terms and waits for you to type `I
agree to the WU-Minn HCP Open Access Data Use Terms`; only then does it
download and MD5-verify
[Zenodo 10.5281/zenodo.20736221](https://doi.org/10.5281/zenodo.20736221).
It never accepts on your behalf, and it refuses rather than blocks when stdin
is not a terminal, so run it interactively. Already have the zip? Extract it
to `~/.local/share/glow/hcp100_dki_noddi_mni` and the call is a no-op. The WGN
arm needs no data.

## Cache and records

Every build and every measurement is memoised twice over: joblib keeps the
heavy Python object -- a cell's payload, a fitted score -- under
`~/.local/share/glow/cache`, and a parallel tree of one-JSON-per-call records
under `~/.local/share/glow/records` holds that same call's inputs, outputs and
timing under the same key.

Those records link into a DAG, because each one names its parents: a cell
build (`get_exp_effect`) feeds the measurements that score it. A **leaf** is
a record nothing else consumes -- one method fitted and scored on one cell --
and it is the unit every count in this file is in.

The cache is disposable and the records are not;
`glow/_extra/benchmark/recorder.py` has the whole scheme. Ours: the joblib
cache is 1.8 GB and the records 338 MB.

## The configs

Hours are the recorded leaf `time_sec` summed per cache: wall clock of each
leaf, GLOW's on 10 workers and the GPU, the voxel-wise arms' on one core.
A leaf shared by two caches counts in both.

| config | leaves | h | backs |
|---|---|---|---|
| `sweep_llr` | 8,756 | 652.5 | Fig. 10, and the numbers Section 5.4 quotes (b=1) |
| `vba_tune` | 7,500 | 77.0 | Tab. 2, the kernel width and statistic every voxel-wise arm runs |
| `vba_stat` | 7,500 | 47.0 | Tab. 3 |
| `prune` | 8,712 | 26.0 | Fig. 8; its memoised fits back Fig. 11 |
| `null` | 2,000 | 20.9 | Fig. 9 |
| `runtime_num_vox` | 108 | 1.3 | Fig. 12 |
| `segment` | 3,267 | 0.2 | Fig. 7 |
| `runtime_1perm_*` (4 of 5) | 81 | 0.1 | Fig. 13 |
| `sweep_extent` | 5,952 | 388.1 | not in the paper (358.1 h not shared with `sweep_llr`) |
| `sweep_b` | 1,596 | 143.5 | not in the paper (82.2 h not shared) |
| `sweep_n_perm_inner` | 1,029 | 5.7 | settles `N_PERM_INNER`; not drawn in the paper |

Plus 116.6 h of recorded cell builds (`get_exp_effect`). Three heavy
intermediates are memoised but not recorded, the largest being
`glow_fit_for_prune`, so a cold run costs more than this.

## Running

The whole catalogue, in lanes that do not contend for the device:

```bash
glow/_extra/benchmark/config_run_all.sh
```

It runs every cache above plus `smoke` and `sweep_llr_wgn_sphere` (1,100
cells, GPU-bound, not in the paper), and not `oracle_stat`. One cache at a
time:

```bash
python -m glow._extra.benchmark --list             # the catalogue
python -m glow._extra.benchmark sweep_llr          # GPU where visible
python -m glow._extra.benchmark -j 4 --no-gpu sweep_llr   # CPU, parallel
python -m glow._extra.benchmark --method VBA --method CET null
```

- **Resumable by default.** A run computes only what the records do not hold,
  skipping a finished cell outright and, within an unfinished one, the leaves
  already recorded. `--no-skip` forces the whole grid.
- **`GLOW_BENCH_N_SEED=5`** caps every seed grid, giving a complete
  catalogue in every other axis at a fraction of the cost. The cap never
  enters a cell's identity, so a later full run reuses what it produced. It
  caps `null`'s 1000 seeds too, so never set it when running `make_csv` or
  `plot` (plot warns if it is set).

## From records to figures

Running only fills the records; turning them into figures is separate.

```bash
python -m glow._extra.benchmark.make_csv     # one tidy CSV per cache
python -m glow._extra.benchmark.plot         # figures + tables
```

CSVs land in `~/.local/share/glow/results/<cache>.csv` and figures in
`~/.local/share/glow/results/_latest/<cache>/`. The paper's files:

| paper | file under `_latest/` |
|---|---|
| Fig. 7 | `segment/segment.pdf` |
| Fig. 8 | `prune/prune_Focus.pdf` |
| Fig. 9 | `null/null_calibration.pdf` |
| Fig. 10 | `sweep_llr/sweep_llr_b1.pdf` |
| Fig. 12 | `runtime_num_vox/runtime_num_vox_runtime.pdf` |
| Fig. 13 | `runtime_1perm/runtime_1perm_grid.pdf` |
| Tab. 2 | `vba_tune/fwhm_dice.tex` |
| Tab. 3 | `vba_stat/stat_dice.tex` |

Fig. 13's grid is drawn only when its four `runtime_1perm_*` caches are
plotted in one call (as `'runtime*'` or no argument does). Fig. 11 needs more
than the records; see [below](#figures-the-records-alone-cannot-draw).

## Using our records

The published records hold every experiment the paper reports, and only
those: each paper cache's leaves (`sweep_llr` at b = 1 alone) and the cell
records they descend from, 36,941 JSON files, exported by

```bash
python -m glow._extra.benchmark.scripts.paper_records --out <dir>
```

To redraw the paper from them, with no compute and without the imaging data:

```bash
mkdir -p ~/.local/share/glow
unzip glow-paper-records.zip -d ~/.local/share/glow    # -> .../glow/records/
python -m glow._extra.benchmark.plot segment prune null sweep_llr \
    vba_tune vba_stat 'runtime*'
```

(`plot` with no argument draws these too, and also partial `sweep_extent` and
`sweep_b` figures from the few cells those caches share with `sweep_llr`;
neither is in the paper.) Every table this draws is byte-identical
to ours and every figure identical to ours in its text and pixels. The
deposit's `manifest.json` counts its records per cache.

With them in place a benchmark run also skips every cell they cover. The
records and the joblib cache are independent halves: the records are enough
for the figures, not enough to resume a fit.

## Checking our records

To check that a record is what its recipe computes, recompute it. This draws
the deposit's leaves in a seeded random order and recomputes as many as fit
in a wall-clock budget, each cell realized from scratch and each leaf refit,
then diffs every output against its record:

```bash
python -m glow._extra.benchmark.scripts.validate_records \
    --seed 0 --runtime 2h --records glow-paper-records.zip
```

The paper caches take turns, so even a short budget reaches every figure;
`--cache` and `--source wgn` narrow the draw (WGN needs no imaging data), and
`--dry-run` prices each cache without running anything. It never touches the
cache or the records. Each leaf comes back `identical`, `close` (counts
equal, floats within `--rtol`) or `differs`, logged one JSON line per leaf
under `~/.local/share/glow/results/validate_records/`; the exit status is 1
if any differs or fails. At `2f0c04e` on our machine expect `identical`
bar the few leaves [below](#how-we-checked-ours); on another machine, or at
a later commit (see [Two seeds that missed their
plant](#two-seeds-that-missed-their-plant)), expect the drift
[below](#why-it-is-not-bitwise-reproducible). The
timing leaves are judged on their voxel count, and their time ratio is
logged.

### How we checked ours

Before publishing version 2 we ran this check against the zip as deposited
(md5 `38c7c80a3b4bee994c419deabe5b5ad1`), on the machine that computed the
records. The code under test was a `git archive` of `2f0c04e` on
`PYTHONPATH`, so no working-tree edit could reach it. Ten lanes, each a
`validate_records` loop over its own `--seed` and `--cache`, ran for 63 hours
from 2026-10-02. Separately, the paper was redrawn from the zip alone in a
fresh `XDG_DATA_HOME` and compared with the manuscript's copies.

| cache | leaves | recomputed | identical | differs |
|---|---:|---:|---:|---:|
| `null` | 2,000 | 322 | 322 | 0 |
| `prune` | 8,712 | 5,128 | 5,128 | 0 |
| `segment` | 3,267 | 1,925 | 1,925 | 0 |
| `sweep_llr` | 4,356 | 369 | 368 | 1 |
| `vba_stat` | 7,500 | 6,870 | 6,866 | 4 |
| `vba_tune` | 7,500 | 1,641 | 1,641 | 0 |
| the five runtime caches | 189 | 189 | 189 | 0 |
| **all** | **33,524** | **16,444 (49%)** | **16,439** | **5** |

Counts are distinct leaves. 3,283 were recomputed more than once and gave
the same verdict every time; none came back `close` or failed. What came up:

- **Five leaves differ, all one variant**: VBA-TFCE on z-scored Wilks'
  Lambda, which is also Fig. 10's VBA-TFCE arm. In each, one null draw
  crosses the observed peak. Four move `min_pval` by exactly 1/501, in both
  directions; the fifth (`9d6f0586c54f`) drops one false-positive voxel near
  p = 0.05 (fp 4 -> 3). Every other statistic variant replayed identically,
  LLR and Pillai on the same walks included. Our leading explanation, not
  proven since the old library is gone: the records predate the host's
  2026-09-30 move to glibc 2.43; Wilks passes through `np.expm1`, which here
  is glibc's and not correctly rounded; and FSL's float32 TFCE height step
  turns that 1-ulp change into a jump in one draw's maximum. No paper number
  moves. No figure or table draws `min_pval` outside the GLOW null cache,
  and the deposit patched with the replayed scores of the two leaves under
  a paper figure or count (`257e0ebe2154` in `sweep_llr`, `9d6f0586c54f` in
  `vba_stat`) redraws Fig. 10 pixel-identical and `stat_dice.tex`
  byte-identical.
- **One table does not redraw at `2f0c04e`.** 21 of the redraw's 22 checks
  pass: every figure pixel-identical, every figure CSV and `detection_stats`
  byte-identical. The exception is the paper's kernel width by effect
  strength table (`vba_tune/fwhm_llr_dice.tex`), whose generator is commit
  `b078418`, not on `main`. Plotting `vba_tune` from the deposit at
  `b078418` reproduces it byte-identical. The leaf verdicts above stand,
  since no lane imports the plot layer.
- **`vba_tune` recomputes slowly.** The validator visits a cell's sibling
  leaves in random order, but a `vba_tune` cell's 150 leaves span five
  kernel widths, each its own voxel walk of about ten minutes, and only one
  walk is memoised. Nearly every leaf pays for a walk (about 7 leaves an
  hour per lane), which is why its coverage trails. Give it several
  single-thread lanes, as we did, or expect days.

## One CSV per figure

```bash
python -m glow._extra.benchmark.scripts.figure_csv --out-dir figure_csv \
    --oracle-csv <oracle_vs_k.csv>
```

It writes twice: to `--out-dir`, and to `figure_csv/` inside the records
tree, so the CSVs travel with the records they came from (`--no-records-copy`
skips the second). `--oracle-csv` is the curves `oracle_vs_k.py` computed;
without it, the copy already beside the records is used. The published
records carry the set.

| paper | csv | rows | cols |
|---|---|---|---|
| Fig. 7 segmentation quality | `fig07_segment.csv` | 3,267 | 15 |
| Fig. 8 pruning rule (Focus) | `fig08_prune_focus.csv` | 4,356 | 17 |
| Fig. 9 FWER control | `fig09_null_calibration.csv` | 2,000 | 21 |
| Fig. 10 detection vs LLR (b=1) | `fig10_sweep_llr_b1.csv` | 4,356 | 21 |
| Fig. 11 region-budget oracle | `fig11_oracle_vs_k.csv` | 5,410 | 9 |
| Fig. 12 wall clock vs num_vox | `fig12_runtime_num_vox.csv` | 108 | 6 |
| Fig. 13a per-perm vs num_vox | `fig13a_runtime_1perm_num_vox.csv` | 27 | 6 |
| Fig. 13b per-perm vs n_perm_inner | `fig13b_runtime_1perm_n_perm_inner.csv` | 21 | 6 |
| Fig. 13c per-perm vs b | `fig13c_runtime_1perm_b.csv` | 18 | 6 |
| Fig. 13d per-perm vs num_img | `fig13d_runtime_1perm_nimg.csv` | 15 | 6 |
| Tab. 2 kernel-width tuning | `tab02_fwhm_dice.csv` | 7,500 | 9 |
| Tab. 3 statistic check (b=2) | `tab03_stat_dice.csv` | 7,500 | 8 |

4.3 MB for the set. Figures 1-4 are schematics and 5-6 are effect
illustrations (`make_effect_figs.py` beside the manuscript, which needs the
imaging data), so they have no tabular data.

The grain is per trial, not per plotted point: a figure shows a mean and a
band, and the CSV under it holds the trials those summarise, every row
carrying its `seed`. The exporter calls the plot layer's own tidy functions
and row filters, so a CSV is the frame its figure was drawn from, and sorts
its rows, so two runs are byte-identical and a CSV can be diffed against
ours.

## Figures the records alone cannot draw

A record carries its leaf's score, not the fitted object, so a figure reading
a Ward tree or a significant set needs more. Only `oracle_vs_k` (Fig. 11)
does. It caches its per-trial curves to a CSV beside the figure it writes,
and redraws from that CSV with no compute:

```bash
cp ~/.local/share/glow/records/figure_csv/fig11_oracle_vs_k.csv .
python -m glow._extra.benchmark.scripts.oracle_vs_k --out fig11_oracle_vs_k.pdf
```

`--recompute` instead rescores the 297 cells from the memoised
`glow_fit_for_prune` fits, which are not part of the published records;
lacking them it raises rather than refit silently, and
`python -m glow._extra.benchmark prune` is what rebuilds them.

## Numbers quoted in the text

Every number the Detection Performance section states that no figure draws
(the seed-paired Dice averages, the sensitivity gain, where GLOW's false
volume sits, completeness as a one-region rate, region purity, and the split
of the precision gap between prune rule, test and segmentation) prints from
the records and Fig. 11's CSV:

```bash
python -m glow._extra.benchmark.scripts.detection_stats \
    --oracle-csv ~/.local/share/glow/records/figure_csv/fig11_oracle_vs_k.csv
```

## Why it is not bitwise reproducible

The recipes, the seeds and the decisions reproduce; the floating-point bytes
do not. Bitwise identical on any machine: every artifact's identity (a recipe
uid is sha256 over canonical JSON, no array bytes), all seeding (every draw
comes from `np.random.default_rng(seed)`, outer permutation *k* seeded by
*k*), the clean Experiments, the planted effect's support, and the Ward trees
given identical input.

What is not:

1. **`impose_effect` is the first call that touches BLAS**, and numpy ships
   OpenBLAS with `DYNAMIC_ARCH`, so the gemm micro-kernel is chosen from the
   host CPU. A different kernel moves the planted intensities by ~1e-07
   relative and every statistic downstream with them. We ran the Haswell
   kernel, on an i9-14900K (AVX2, no AVX-512).
2. **The GLOW arm's device and dtype come from the machine.** Ours was CUDA
   float32 on an RTX 4060; a CPU-only box runs float64 and lands ~2e-07
   relative away in z. Each leaf records which it used, under
   `inputs.fit_params`.
3. **The TFCE arm's backend comes from the machine.** Ours was FSL 6.0.7.16,
   selected because `fslmaths` was installed. The python fallback uses a
   different height grid and differs by about 3% at the top step.
4. Smaller: LAPACK eigenvector signs are not fixed by the standard, and
   `Recorder.load` reads its JSON unsorted, so raw CSV row order follows your
   filesystem.

## Two seeds that missed their plant

Every rebuild re-measures its plant (`cell.ExpEffect._check_effects`), and
a cell whose per-voxel LLR misses its target by more than `cell.LLR_RTOL`
(0.1%) is skipped. At `2f0c04e`, 22 HCP b = 1 cells miss and have no
records: 20 of seed 12 (every `sweep_llr` strength, plus `sweep_extent`) and
2 of seed 15 (`sweep_extent` only). Among the paper's caches that drops
seed 12 from `sweep_llr`, `segment` and `prune`, which report 49 HCP seeds;
seed 15 is in none of them.

Those plants really were 0.1-0.8% off target, and the cause is float32
arithmetic, not the solver (which meets its own constraint to ~1e-15).
`compute_offset` forms the region's scatter as a raw sum of squares less
num_vox * mean^2, so float32 rounding in that sum is amplified by
(mean / sd)^2:

- **Seed 12** draws `icvf`, and its support lies in white matter saturated
  at the NODDI ceiling: mean 0.92, mean / sd ~ 14, 19% of its values exactly
  0.99. Summing that many identical values biases the float32 total one way,
  and the plant lands 0.2-0.3% weak at every strength.
- **Seed 15**'s region mean is nearly orthogonal to the interest regressor,
  so the solve scales that projection ~600x, its float32 rounding included.

The current code solves (`compute_offset`, `impose_effect`) and checks in
float64, which plants both seeds within 1e-6 of target; `2f0c04e` did both in
float32. What that difference touches:

- **An existing store.** A cell's payload holds its solved offset, so a cell
  already realized keeps it: no recorded number changes, and every cached
  cell passes the float64 check (all 3,944 of ours re-measured; largest miss
  8.0e-4).
- **A run from scratch.** A fresh cell solves a slightly different offset
  (~1e-6 of the intensity scale), which moves the float32 bytes of most
  planted voxels, so recomputed leaves drift from the version 2 records as
  they do [across machines](#why-it-is-not-bitwise-reproducible).
  `validate_records` reproduces those records bit for bit only at `2f0c04e`.

To fill the holes, move the 22 cells' cache entries aside, so every one is
re-solved in float64, then rerun the paper caches; `drive` re-realizes a
cell whose entry is gone, and a re-realized cell overwrites its record:

```bash
python -c "
import shutil
from glow._extra.benchmark.config import CONFIG
from glow._extra.benchmark.cell import exp_effect_recipe, get_exp_effect
from glow._extra.benchmark.file import get_path_cache
from glow._extra.benchmark.recorder import Recorder
HOLES = {'a1eb5daf', '36320490', '8b74cfac', '488f0a61', 'ab115e4c',
         '2c56928a', 'e80bab4c', '08a8ff14', 'ff6b54e2', '260cb237',
         '16756818', 'e2840594', '8f088f73', '34e90b2a', '55225a14',
         'babd096d', 'ae8cbf31', '3dc1e4fd', '9f2d9e67', 'b401a95e',
         'f5e7623c', '9371ca97'}
src = get_path_cache() / 'glow/_extra/benchmark/cell/get_exp_effect'
dst = get_path_cache().parent / 'old/plant_holes_float32'
dst.mkdir(parents=True, exist_ok=True)
for data_grid, effect_grid, *_ in CONFIG.values():
    for kd in data_grid:
        for ke in effect_grid:
            if ke is None or exp_effect_recipe(kd, ke).uid[:8] not in HOLES:
                continue
            key = Recorder._args_hash(get_exp_effect.func, (kd, ke), {})
            if (src / key).is_dir():
                shutil.move(src / key, dst / key)"
python -m glow._extra.benchmark sweep_llr segment prune
```
