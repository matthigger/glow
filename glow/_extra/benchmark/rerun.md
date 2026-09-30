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
| input | this repository | [github.com/matthigger/glow](https://github.com/matthigger/glow); computation at `cc7e53e`, read with the tree this file ships in |
| input | HCP-YA imaging data, 100 subjects x 6 maps | [Zenodo 10.5281/zenodo.20736221](https://doi.org/10.5281/zenodo.20736221) |
| output | provenance records, version 2 | [Zenodo 10.5281/zenodo.22664285](https://doi.org/10.5281/zenodo.22664285) |

Version 1 of the records DOI holds an earlier corpus, computed at `25f528c`
on a cache layer that keyed artifacts partly on an Experiment's bytes. That
layer is gone: a cell is now declared, stored as a small payload and rebuilt
per leaf (`glow/_extra/benchmark/cell.py`), so no record of version 1 is
addressable from this code, and its plants were drawn from different seeds
(seed 7 there is not seed 7 here). Version 2 is the corpus the paper reports.

### The corpus in `~/.local/share/glow`

A record does not name the commit that computed it, so the commit is pinned
here. A file cannot name its own commit either, so what is pinned is the last
commit that changes what the reported caches compute. Commits on top of it
are documentation, add caches, or change how records are read, and move no
recorded number.

| | |
|---|---|
| computation pinned at | `cc7e53e` (the tree the 50-seed corpus ran from) |
| branch | `main` |
| seeds | 50 (`null` 1000; `vba_tune`, `oracle_stat` 10) |
| tuned | 1 - Wilks at 2 mm for VBA, VBA-TFCE, CET; z for VBA-TFCE only |
| holes | 22 HCP b=1 cells (seeds 12, 15) fail the plant check (`cell.LLR_RTOL`) and are skipped |
| not in the paper | `sweep_llr_wgn_sphere` (declared at `d3cfce8`), `sweep_extent`, `sweep_b`, `sweep_n_perm_inner`, `oracle_stat`, `smoke`, `runtime_1perm_n_perm_fwer` |
| old cache | `~/.local/share/glow/cache.bak_prepayload_20260917` |
| old records | `~/.local/share/glow/records.bak_prepayload_20260917` |

The leaves were computed in stages at different commits. Each stage is
equivalent to `cc7e53e` for the leaves it left in the store:

| leaves | computed | at | why it matches `cc7e53e` |
|---|---|---|---|
| seeds 0-9, every cache | 2026-09-17 to 09-19 | `8880bf8` to `21c3fe4` | only benchmark declarations changed in that range, and a leaf is keyed on its declared recipe, so a stale one is unreferenced rather than read |
| `vba_tune` | 2026-09-17 | `1151296` | the width and statistic it chose are what `d7ca049` wired in |
| seeds 10-49 | 2026-09-19 to 09-25 | `21c3fe4`, then `21c3fe4` + the `cc7e53e` diff | `cc7e53e` drops Oracle-RBA and skips unbuildable cells; neither moves another arm's number |
| WGN VBA, VBA-TFCE, CET, every seed | 2026-09-25 to 09-26 | `cc7e53e`'s tree | recomputed after `5c15b8e` put WGN on a 2 mm grid; the 2-voxel-kernel leaves are in `~/.local/share/glow/old/wgn_affine_20260925/` |

`5c15b8e` touches only the affine, which only `Experiment.smooth` reads, so
GLOW and every HCP leaf are unaffected by it. The cached WGN cell payloads
were patched in place to carry the affine (masks and offsets unchanged;
originals in the same `old/` directory).

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

`cc7e53e` and later report only the holes: sweep_llr 11, sweep_extent 12,
sweep_b 1, segment 11, prune 11, sweep_n_perm_inner 3, and 0 for null,
vba_stat and vba_tune -- 22 distinct cells in all.

That check sees every change to a declared recipe or cell. It cannot see a
change to code that runs underneath one without entering its identity (the
affine was such a change); that needs a leaf recomputed and its score
compared. Three voxel-wise `sweep_llr` leaves (WGN VBA, WGN CET, HCP
VBA-TFCE) re-fit on CPU at this tree reproduce their recorded confusion
counts, minimum p-value and region count exactly.

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

With them in place a benchmark run also skips every cell they cover, which is
how to recompute a subset and diff it against ours. The records and the
joblib cache are independent halves: the records are enough for the figures,
not enough to resume a fit.

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
