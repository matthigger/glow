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
| input | this repository | [github.com/matthigger/glow](https://github.com/matthigger/glow), at `25f528c` |
| input | HCP-YA imaging data, 100 subjects x 6 maps | [Zenodo 10.5281/zenodo.20736221](https://doi.org/10.5281/zenodo.20736221) |
| output | provenance records | [Zenodo 10.5281/zenodo.22664285](https://doi.org/10.5281/zenodo.22664285) |

You do not have to generate the output to read it. Ours is published, so the
figures redraw from it with no compute and without the imaging data -- see
[Using our records](#using-our-records), or [One CSV per
figure](#one-csv-per-figure) if you only want the plotted numbers. Generating
your own is what the rest of this file is for; it is worthwhile to note [why it is not bitwise
reproducible](#why-it-is-not-bitwise-reproducible).

## Install

```bash
git clone https://github.com/matthigger/glow && cd glow
python -m venv ~/venv_glow
~/venv_glow/bin/pip install -r requirements-lock.txt
~/venv_glow/bin/pip install -e . --no-deps
```

`requirements-lock.txt` pins every package in the environment that produced
the records. Optional: **FSL** (the TFCE arm uses `fslmaths` where it is
installed) and an NVIDIA card (the GLOW arm uses it where it is visible).

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
to
`~/.local/share/glow/hcp100_dki_noddi_mni` and the call is a no-op. The WGN
arm needs no data.

## Cache and records

Every build and every measurement is memoised twice over: joblib keeps the
heavy Python object -- an Experiment, a fitted score -- under
`~/.local/share/glow/cache`, and a parallel tree of one-JSON-per-call records
under `~/.local/share/glow/records` holds that same call's inputs, outputs and
timing under the same key.

Those records link into a DAG, because each one names its parents: a data
build feeds a planted effect, which feeds the measurement that scores it. A
**leaf** is a record nothing else consumes -- one method fitted and scored on
one cell -- and it is the unit every count in this file is in.

The cache is disposable and the records are not;
`glow/_extra/benchmark/recorder.py` has the whole scheme.

## The configs the paper reads

| config | CPU-h | backs |
|---|---|---|
| `sweep_llr` | 701.9 | detection vs effect strength |
| `sweep_extent` | 451.7 | detection vs effect extent |
| `vba_stat` | 185.6 | MANCOVA-statistic table |
| `sweep_b` | 147.0 | detection vs feature count |
| `null` | 29.8 | FWER calibration |
| `sweep_n_perm_inner` | 5.8 | settles `N_PERM_INNER` |
| `prune` | 0.5 | pruning-rule selection |
| `segment` | 0.2 | segmentation quality |
| `runtime_num_vox` | 1.3 | wall clock vs volume |
| `runtime_1perm_*` (5) | 0.1 | per-permutation cost |

The CPU-hours are what the records recorded, which understates a cold run:
three heavy intermediates are memoised but not recorded, the largest being
`glow_fit_for_prune`. Disk: the joblib cache reached 104 GB.

## Running

The whole catalogue, in lanes that do not contend for the device:

```bash
glow/_extra/benchmark/config_run_all.sh
```

One cache at a time:

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
  enters a cell's identity, so a later full run reuses what it produced.

## From records to figures

Running only fills the records; turning them into figures is separate.

```bash
python -m glow._extra.benchmark.make_csv     # one tidy CSV per cache
python -m glow._extra.benchmark.plot         # figures + tables
```

Output lands in `~/.local/share/glow/results/_latest/<cache>/`.

## Using our records

Download the record tree from
[Zenodo 10.5281/zenodo.22664285](https://doi.org/10.5281/zenodo.22664285),
unpack it into `~/.local/share/glow/records/`, then
`python -m glow._extra.benchmark.plot` redraws the paper's figures with no
compute and without the imaging data. With them in place a benchmark run also
skips every cell they cover, which is how to recompute a subset and diff it
against ours.

The records and the joblib cache are independent halves: the records are
enough for the figures, not enough to resume a fit.

## One CSV per figure

```bash
python scripts/figure_csv.py --out-dir figure_csv
```

It writes twice: to `--out-dir`, and to `figure_csv/` inside the records
tree, so the CSVs travel with the records they came from.

| figure | csv | rows | cols |
|---|---|---|---|
| Fig. 7 segmentation quality | `fig07_segment.csv` | 3,300 | 15 |
| Fig. 8 pruning rule (Focus) | `fig08_prune_focus.csv` | 4,400 | 17 |
| Fig. 9 FWER control | `fig09_null_calibration.csv` | 2,000 | 21 |
| Fig. 10 detection vs LLR (b=1) | `fig10_sweep_llr_b1.csv` | 4,400 | 21 |
| Fig. 11 region-budget oracle | `fig11_oracle_vs_k.csv` | 5,470 | 9 |
| Fig. 12 wall clock vs num_vox | `fig12_runtime_num_vox.csv` | 108 | 6 |
| Fig. 13a per-perm vs num_vox | `fig13a_runtime_1perm_num_vox.csv` | 27 | 6 |
| Fig. 13b per-perm vs n_perm_inner | `fig13b_runtime_1perm_n_perm_inner.csv` | 21 | 6 |
| Fig. 13c per-perm vs b | `fig13c_runtime_1perm_b.csv` | 18 | 6 |
| Fig. 13d per-perm vs num_img | `fig13d_runtime_1perm_nimg.csv` | 15 | 6 |
| Tab. 2 statistic bake-off | `tab02_stat_dice.csv` | 16,500 | 8 |

4.8 MB for the set. Figures 1-4 are schematics and 5-6 are effect
illustrations, so they have no tabular data.

The grain is per trial, not per plotted point: a figure shows a mean and a
band, and the CSV under it holds the trials those summarise, every row
carrying its `seed`. The exporter calls the plot layer's own tidy functions
and row filters, so a CSV is the frame its figure was drawn from, and sorts
its rows, so two runs are byte-identical and a CSV can be diffed against
ours.

## Figures the records alone cannot draw

A record carries its leaf's score, not the fitted object, so a figure reading
a Ward tree or a significant set needs more. Only `oracle_vs_k` (Fig. 11)
does.

```bash
python scripts/oracle_vs_k.py --out fig11.pdf
```

It reads the per-trial curves cached in `oracle_vs_k.csv` beside the output
(410 KB, shipped) and redraws with no compute. `--recompute` instead rescores
the 300 cells from the memoised `glow_fit_for_prune` fits, which are not part
of the shipped records; lacking them it raises rather than refit silently, and
`python -m glow._extra.benchmark prune` is what rebuilds them.

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
