"""Bake a representative set of the paper's cells for the hosted viewer.

Each entry in DEMOS names one cell of the benchmark catalogue -- a point on
an axis some paper figure sweeps -- and this module builds it, fits the
paper's GLOW recipe on it, and writes {ana, exp, mask_target, demo} to a
gzipped pickle. exp is bundled because the analysis does not store it and
the viewer needs it (see glow.analysis._glow's __getstate__).

Usage:
    python -m glow._extra.viewer.web.bake_demos              # bake all
    python -m glow._extra.viewer.web.bake_demos --force      # refit
    python -m glow._extra.viewer.web.bake_demos --only KEY   # one entry
    python -m glow._extra.viewer.web.bake_demos --list       # plan only

Every axis value is read off glow._extra.benchmark.config rather than
retyped, so a demo is the same cell the corresponding figure reports and
moves with the paper's grid. The analysis is the shipped recipe
(config.REPORTED_GLOW_LABEL) with only the knob a demo varies overridden.

Cells are built through the undecorated factories (call_uncached), so the
shared benchmark cache is neither read nor written and no provenance
record is filed: an ad-hoc build landing on a catalogue cell's key would
rewrite that record with a fresh exp hash and drop every finished leaf
that consumed the old one out of config_results_df.

WGN only by default. HCP entries are gated behind --hcp: the maps are
DUA-restricted, so an HCP-derived bundle is fine to view locally and not
fine to publish. See docs/notes for the resampling that would lift that.

The run also writes manifest.json, which the server reads at boot so the
landing page can list the set without unpickling any of it.
"""

import argparse
import dataclasses
import gzip
import inspect
import json
import pathlib
import pickle
import sys
import time

from glow._extra.benchmark import config
from glow._extra.benchmark.data import data_factory, data_recipe
from glow._extra.benchmark.data import effect_factory
from glow._extra.viewer.web import MANIFEST_NAME
from glow.analysis import AnalysisGLOW
from glow.analysis.cluster import ClusterMode

# the recipe every demo starts from: the arm the paper reports.
_PAPER_ANA = config.ana_kwargs_dict[config.REPORTED_GLOW_LABEL]

# the effect-strength axis, by the name a reader uses for it. The endpoints
# and midpoint of the sweep the power curve is read off.
_LLR_WEAK = float(config.EFFECT_LLR_GRID[0])
_LLR_MODERATE = config.MODERATE_EFFECT_LLR
_LLR_STRONG = float(config.EFFECT_LLR_GRID[-1])

# the extent axis endpoints: the narrowest support on the grid and the whole
# analysis volume.
_EXTENT_NARROW = float(config.EXTENT_FRAC_GRID[0])
_EXTENT_WHOLE = float(config.EXTENT_FRAC_GRID[-1])


@dataclasses.dataclass(frozen=True)
class Demo:
    """One baked cell: which catalogue point it is and how to fit it.

    Attributes:
        key (str): filename stem and URL segment; stable across rebuilds.
        blurb (str): one-line reader-facing label for the landing page.
        cache (str): the CONFIG cache whose axis this cell sits on, for
            the landing page to group by.
        data (dict): config.DATA_AXES overrides, each a single-value list
            so the grid it builds holds exactly one cell.
        effect (dict): config.EFFECT_AXES overrides; llr_list=None is the
            null path.
        ana (dict): AnalysisGLOW knob overrides on the reported recipe.
    """

    key: str
    blurb: str
    cache: str
    data: dict = dataclasses.field(default_factory=dict)
    effect: dict = dataclasses.field(default_factory=dict)
    ana: dict = dataclasses.field(default_factory=dict)

    @property
    def source(self) -> str:
        """Return the demo's image source ('wgn' or 'hcp')."""
        return self.data.get('sources', ['wgn'])[0]


# One entry per axis a reader would want to move, not one per cell the
# catalogue holds: the paper's grids run to thousands of cells, and a
# hosted set is read by clicking through it. Every entry is WGN at the
# catalogue's default geometry unless it is the entry that varies that.
#
# The moderate-effect Focus fit is the hub: the llr, extent, projection
# and prune entries are all one step off it, so a reader can hold it in
# mind and see what one axis does.
DEMOS = [
    # --- the power curve (sweep_llr, b=1) ----------------------------
    Demo(key='llr_weak', cache='sweep_llr',
         blurb='Weakest effect on the strength grid',
         effect=dict(llr_list=[_LLR_WEAK])),
    Demo(key='llr_moderate', cache='sweep_llr',
         blurb='Moderate effect -- the anchor the other sweeps plant',
         effect=dict(llr_list=[_LLR_MODERATE])),
    Demo(key='llr_strong', cache='sweep_llr',
         blurb='Strongest effect on the strength grid',
         effect=dict(llr_list=[_LLR_STRONG])),

    # --- no effect (null) --------------------------------------------
    Demo(key='null', cache='null',
         blurb='No planted effect -- the FWER calibration case',
         effect=dict(llr_list=None)),

    # --- feature count (sweep_b) -------------------------------------
    Demo(key='b2', cache='sweep_b',
         blurb='Two imaging features, moderate effect',
         data=dict(b_list=[2])),
    Demo(key='b4', cache='sweep_b',
         blurb='Four imaging features, moderate effect',
         data=dict(b_list=[4])),

    # --- effect extent (sweep_extent) --------------------------------
    Demo(key='extent_narrow', cache='sweep_extent',
         blurb='Narrowest support on the extent grid',
         effect=dict(n_vox_frac_list=[_EXTENT_NARROW])),
    Demo(key='extent_whole', cache='sweep_extent',
         blurb='Effect covering the whole analysis volume',
         effect=dict(n_vox_frac_list=[_EXTENT_WHOLE])),

    # --- Ward projection (segment) -----------------------------------
    # the third mode, Focus, is llr_moderate above.
    Demo(key='ward_naive', cache='segment',
         blurb='Ward on raw y (Naive projection)',
         ana=dict(cluster_mode=ClusterMode.NAIVE)),
    Demo(key='ward_glm_error', cache='segment',
         blurb='Ward on the whole design space (GLM Error projection)',
         ana=dict(cluster_mode=ClusterMode.GLM_ERROR)),

    # --- selection rule (prune) --------------------------------------
    # the same permutation test as llr_moderate, read out by another rule.
    Demo(key='prune_single_max', cache='prune',
         blurb='Single max-LLR region instead of the greedy set',
         ana=dict(prune_rule='single_max')),
    Demo(key='prune_dp', cache='prune',
         blurb='Dynamic-programming cut instead of the greedy set',
         ana=dict(prune_rule='dp')),

    # --- volume (runtime_num_vox) ------------------------------------
    # the one entry fit with keep_stat, which adds the viewer's
    # PERMUTATION panel: the draw matrix is (n_perm_fwer + 1) x num_reg,
    # so only the smallest volume can afford to carry it in a bundle.
    Demo(key='vox_1k', cache='runtime_num_vox',
         blurb='Small volume, with the permutation-draw histogram',
         data=dict(crop_n_vox=config.RUNTIME_NUM_VOX_GRID[0]),
         ana=dict(keep_stat=True)),
    Demo(key='vox_full_brain', cache='runtime_num_vox',
         blurb='Full-brain volume, the top of the runtime sweep',
         data=dict(crop_n_vox=config.RUNTIME_NUM_VOX_GRID[-1])),

    # --- HCP mirrors of the power curve (--hcp; not publishable) -----
    Demo(key='hcp_llr_weak', cache='sweep_llr',
         blurb='HCP diffusion maps, weakest effect',
         data=dict(sources=['hcp']), effect=dict(llr_list=[_LLR_WEAK])),
    Demo(key='hcp_llr_moderate', cache='sweep_llr',
         blurb='HCP diffusion maps, moderate effect',
         data=dict(sources=['hcp']), effect=dict(llr_list=[_LLR_MODERATE])),
    Demo(key='hcp_llr_strong', cache='sweep_llr',
         blurb='HCP diffusion maps, strongest effect',
         data=dict(sources=['hcp']), effect=dict(llr_list=[_LLR_STRONG])),
]


def call_uncached(fnc, *args, **kwargs):
    """Call a benchmark builder with its cache and recorder peeled off.

    The builders in glow._extra.benchmark.data are wrapped @MEMORY.cache
    over @RECORDER; inspect.unwrap walks past both to the raw function,
    which does the same build in memory alone. See the module docstring
    for why a bake must not write either.

    Args:
        fnc: a decorated builder (data_factory / effect_factory reaches
            one by dispatching on source / kind).

    Returns:
        whatever the raw builder returns.
    """
    return inspect.unwrap(fnc)(*args, **kwargs)


def paper_ana(**override) -> AnalysisGLOW:
    """Build the reported GLOW recipe, with the given knobs overridden.

    Reads every knob off config's shipped recipe rather than retyping it,
    so a demo differs from the paper's arm only where it says it does.

    Args:
        **override: any AnalysisGLOW keyword to replace.

    Returns:
        ana (AnalysisGLOW): the recipe to fit.
    """
    kwargs = dict(n_perm_fwer=_PAPER_ANA.n_perm_fwer,
                  n_perm_inner=_PAPER_ANA.n_perm_inner,
                  alpha_fwer=_PAPER_ANA.alpha_fwer,
                  min_vox=_PAPER_ANA.min_vox,
                  cluster_mode=_PAPER_ANA.cluster_mode,
                  prune_rule=_PAPER_ANA.prune_rule,
                  prune_lam=_PAPER_ANA.prune_lam,
                  prune_exp_n_eff=_PAPER_ANA.prune_exp_n_eff)
    kwargs.update(override)
    return AnalysisGLOW(**kwargs)


def build_demo(demo: Demo, *, verbose: bool = True):
    """Build one demo's cell and fit the paper's recipe on it.

    Args:
        demo (Demo): the entry to build.
        verbose (bool): forward progress to the analysis fit.

    Returns:
        ana (AnalysisGLOW): the fitted analysis.
        exp: the Experiment it was fit on, effect included.
        mask_target (np.array | None): the realized (X, Y, Z) bool
            support, or None on the null path.
    """
    # one seed and one source, so the grid each call builds holds exactly
    # the cell this demo names
    data_over = {k: v for k, v in demo.data.items() if k != 'sources'}
    kwargs_data = config.data_grid(seeds=[0], sources=[demo.source],
                                   **data_over)[0]
    kwargs_effect = config.effect_grid(**demo.effect)[0]

    exp = call_uncached(data_factory, **kwargs_data)
    print(f'    y={exp.y.shape} x={exp.x.shape}')

    mask_target = None
    if kwargs_effect is not None:
        kwargs = {k: v for k, v in kwargs_effect.items() if k != 'kind'}
        exp, mask_target_list = call_uncached(
            effect_factory, exp, kind=kwargs_effect['kind'],
            parent_uid=data_recipe(kwargs_data).uid, **kwargs)
        mask_target = mask_target_list[0]
        print(f'    planted {int(mask_target.sum())} voxels')

    ana = paper_ana(**demo.ana)
    print(f'    fitting {ana!r}')
    ana.fit(exp, verbose=verbose, **config.GLOW_FIT_PARAMS)
    print(f'    {len(ana.effect_list)} effect(s) discovered')
    return ana, exp, mask_target


def write_manifest(out_dir: pathlib.Path) -> None:
    """Write manifest.json describing every bundle present in out_dir.

    The server reads this at boot so the landing page can name the set
    without unpickling it; a bundle with no file on disk is left out.

    Args:
        out_dir (pathlib.Path): the bundle directory.
    """
    entries = []
    for demo in DEMOS:
        path = out_dir / f'{demo.key}.p.gz'
        if not path.exists():
            continue
        entries.append({'key': demo.key, 'blurb': demo.blurb,
                        'cache': demo.cache, 'source': demo.source,
                        'size_bytes': path.stat().st_size})
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps({'demos': entries}, indent=2) + '\n')
    print(f'wrote {MANIFEST_NAME} ({len(entries)} bundle(s))')


def main():
    """Bake the selected demos into the output dir as .p.gz bundles."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--out', type=pathlib.Path,
        default=pathlib.Path(__file__).parent / 'pickles',
        help='output directory for baked bundles')
    parser.add_argument(
        '--force', action='store_true',
        help='refit entries whose bundle already exists')
    parser.add_argument(
        '--only', action='append', default=None, metavar='KEY',
        help='bake just this entry (repeatable)')
    parser.add_argument(
        '--hcp', action='store_true',
        help='include the HCP entries (DUA-restricted; do not publish)')
    parser.add_argument(
        '--list', action='store_true',
        help='print the plan and exit without building')
    args = parser.parse_args()

    demo_list = [d for d in DEMOS if args.hcp or d.source != 'hcp']
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {d.key for d in DEMOS}
        if unknown:
            parser.error(f'unknown --only key(s): {sorted(unknown)}')
        demo_list = [d for d in DEMOS if d.key in wanted]

    if args.list:
        for demo in demo_list:
            print(f'{demo.key:<20} {demo.cache:<18} {demo.blurb}')
        print(f'{len(demo_list)} entr(ies)')
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    print(f'baking {len(demo_list)} demo(s) into {args.out}')

    total_t0 = time.time()
    for i, demo in enumerate(demo_list, 1):
        out = args.out / f'{demo.key}.p.gz'
        head = f'[{i}/{len(demo_list)}] {demo.key}'

        if out.exists() and not args.force:
            size_mb = out.stat().st_size / (1024 ** 2)
            print(f'{head}: skip (exists, {size_mb:.1f} MB)')
            continue

        print(f'{head}: {demo.blurb}')
        t0 = time.time()
        ana, exp, mask_target = build_demo(demo)
        build_s = time.time() - t0

        with gzip.open(out, 'wb') as f:
            pickle.dump({'ana': ana, 'exp': exp, 'mask_target': mask_target,
                         'demo': dataclasses.asdict(demo)}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = out.stat().st_size / (1024 ** 2)
        print(f'    -> {out.name} ({size_mb:.1f} MB, {build_s:.0f}s)')
        sys.stdout.flush()

    write_manifest(args.out)
    print(f'done in {time.time() - total_t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
