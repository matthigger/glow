"""Bake a representative set of the paper's HCP cells for the hosted viewer.

Each entry in DEMOS names one cell of the benchmark catalogue -- a point on
an axis some paper figure sweeps -- and this module realizes it, fits the
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

A cell the corpus already realized is read back from the benchmark store,
so the plant a demo shows is the one the figures scored; any other is
realized in memory alone (realize_cell). Nothing is written to the store.

Every entry is HCP, shared under the HCP Open Access Data Use Terms, which
the server asks a visitor to accept before it opens one.

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
import re
import sys
import time

import numpy as np

import glow.mask
from glow._extra.benchmark import config
from glow._extra.benchmark.cell import get_exp_effect
from glow._extra.benchmark.score import score_effects
# the split-VI pair is private to plot, and importing it is still right:
# the alternative is a second copy of a definition the figures own, free
# to drift from them.
from glow._extra.benchmark.plot import _hom_com
from glow._extra.viewer.web import MANIFEST_NAME
from glow.analysis import AnalysisGLOW
from glow.analysis.cluster import ClusterMode

# the one image source the set is drawn from
SOURCE = 'hcp'

# the recipe every demo starts from: the arm the paper reports.
_PAPER_ANA = config.ana_kwargs_dict[config.REPORTED_GLOW_LABEL]

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
            the landing page to group by. A runtime_num_vox entry is drawn
            from that cache's own seed range (demo_cell).
        also (tuple): further caches this same cell sits on. The hub cell
            is a point on several figures' axes at once, and a picker
            that filed it under one of them would offer the others no
            value for it.
        seed_axis (bool): whether --seeds may bake this cell at more than
            one seed. Off for the cell whose bundle and fit are too big to
            hold a seed axis; it stays at seed 0.
        fit (dict): GLOW_FIT_PARAMS overrides for this cell -- how the fit
            runs, never what it computes, so an override cannot change the
            result. Only the largest volume needs one, to stay inside the
            box's RAM.
        data (dict): config.DATA_AXES overrides, each a single-value list
            so the grid it builds holds exactly one cell (crop_n_vox is a
            scalar).
        effect (dict): config.EFFECT_AXES overrides; llr_list=None is the
            null path.
        ana (dict): AnalysisGLOW knob overrides on the reported recipe.
    """

    key: str
    blurb: str
    cache: str
    also: tuple = ()
    seed_axis: bool = True
    fit: dict = dataclasses.field(default_factory=dict)
    data: dict = dataclasses.field(default_factory=dict)
    effect: dict = dataclasses.field(default_factory=dict)
    ana: dict = dataclasses.field(default_factory=dict)


def _llr_demos(also_mid: tuple = ()) -> list:
    """Build one demo per point on the effect-strength grid.

    The whole grid, so the power curve can be walked rather than
    sampled at its ends. The three points the shortcuts name by key keep
    those keys and the rest are numbered by grid index, so a demo already
    linked is never renamed.

    Args:
        also_mid (tuple): further caches the midpoint cell sits on. Only
            the midpoint is a hub -- it is the anchor the other sweeps
            plant.

    Returns:
        list[Demo]: one entry per config.EFFECT_LLR_GRID value.
    """
    n = len(config.EFFECT_LLR_GRID)
    mid = n // 2
    named = {0: 'llr_weak', mid: 'llr_moderate', n - 1: 'llr_strong'}
    told = {0: 'Weakest effect on the grid',
            mid: 'Moderate effect, the anchor the other sweeps plant',
            n - 1: 'Strongest effect on the grid'}

    return [Demo(key=f'hcp_{named.get(i, f"llr_{i:02d}")}',
                 cache='sweep_llr',
                 also=also_mid if i == mid else (),
                 blurb=told.get(i, f'Effect strength {i + 1} of {n}'),
                 effect=dict(llr_list=[float(llr)]))
            for i, llr in enumerate(config.EFFECT_LLR_GRID)]


# One entry per axis a reader would want to move, not one per cell the
# catalogue holds: the paper's grids run to thousands of cells, and a
# hosted set is read by clicking through it. Every entry is at the
# catalogue's default geometry unless it is the entry that varies that.
#
# The moderate-effect Focus fit is the hub: the b, extent, projection and
# prune entries are all one step off it, so a reader can hold it in mind
# and see what one axis does.
DEMOS = [
    # --- the power curve (sweep_llr, b=1) ----------------------------
    *_llr_demos(also_mid=('sweep_b', 'sweep_extent', 'segment', 'prune',
                          'runtime_num_vox')),

    # --- no effect (null) --------------------------------------------
    Demo(key='hcp_null', cache='null',
         blurb='No planted effect -- the FWER calibration case',
         effect=dict(llr_list=None)),

    # --- feature count (sweep_b) -------------------------------------
    # each seed draws its own feature subset (grid.get_kwargs_data_list)
    Demo(key='hcp_b2', cache='sweep_b',
         blurb='Two imaging features, moderate effect',
         data=dict(b_list=[2])),
    Demo(key='hcp_b4', cache='sweep_b',
         blurb='Four imaging features, moderate effect',
         data=dict(b_list=[4])),

    # --- effect extent (sweep_extent) --------------------------------
    Demo(key='hcp_extent_narrow', cache='sweep_extent',
         blurb='Narrowest support on the extent grid',
         effect=dict(n_vox_frac_list=[_EXTENT_NARROW])),
    Demo(key='hcp_extent_whole', cache='sweep_extent',
         blurb='Effect covering the whole analysis volume',
         effect=dict(n_vox_frac_list=[_EXTENT_WHOLE])),

    # --- Ward projection (segment) -----------------------------------
    # the third mode, Focus, is hcp_llr_moderate above.
    Demo(key='hcp_ward_naive', cache='segment',
         blurb='Ward on raw y (Naive projection)',
         ana=dict(cluster_mode=ClusterMode.NAIVE)),
    Demo(key='hcp_ward_glm_error', cache='segment',
         blurb='Ward on the whole design space (GLM Error projection)',
         ana=dict(cluster_mode=ClusterMode.GLM_ERROR)),

    # --- selection rule (prune) --------------------------------------
    # the same permutation test as hcp_llr_moderate, read out by another
    # rule.
    Demo(key='hcp_prune_single_max', cache='prune',
         blurb='Single max-LLR region instead of the greedy set',
         ana=dict(prune_rule='single_max')),
    Demo(key='hcp_prune_dp', cache='prune',
         blurb='Dynamic-programming cut instead of the greedy set',
         ana=dict(prune_rule='dp')),

    # --- volume (runtime_num_vox) ------------------------------------
    # the one entry fit with keep_stat, which adds the viewer's
    # PERMUTATION panel: the draw matrix is (n_perm_fwer + 1) x num_reg,
    # so only the smallest volume can afford to carry it in a bundle.
    Demo(key='hcp_vox_1k', cache='runtime_num_vox',
         blurb='Small volume, with the permutation-draw histogram',
         data=dict(crop_n_vox=config.RUNTIME_NUM_VOX_GRID[0]),
         ana=dict(keep_stat=True)),
    # n_jobs is cut here alone: at the full HCP support each worker holds
    # its own tail fold, and the shared count exhausts this box's RAM.
    Demo(key='hcp_vox_full_brain', cache='runtime_num_vox',
         seed_axis=False, fit=dict(n_jobs=6),
         blurb='The whole brain, the top of the runtime sweep',
         data=dict(crop_n_vox=config.RUNTIME_NUM_VOX_GRID[-1])),
]

_DEMO_BY_KEY = {d.key: d for d in DEMOS}


def demo_key(demo: Demo, seed: int) -> str:
    """Name the bundle for one demo at one seed.

    Seed 0 keeps the bare key: it is the seed the set has always baked,
    and suffixing it would break every link already shared.

    Args:
        demo (Demo): the entry.
        seed (int): the data seed it was built at.

    Returns:
        str: the filename stem and URL segment.
    """
    return demo.key if seed == 0 else f'{demo.key}_s{seed}'


def parse_bundle_name(stem: str):
    """Recover (demo, seed) from a bundle's filename stem.

    Args:
        stem (str): a bundle filename with its .p.gz suffix removed.

    Returns:
        tuple | None: (Demo, seed), or None when the stem names no entry
            in DEMOS -- a bundle from an older set, or dropped in by hand.
    """
    match = re.fullmatch(r'(.+)_s(\d+)', stem)
    if match and match.group(1) in _DEMO_BY_KEY:
        return _DEMO_BY_KEY[match.group(1)], int(match.group(2))
    if stem in _DEMO_BY_KEY:
        return _DEMO_BY_KEY[stem], 0
    return None


def bake_plan(demo_list, seeds) -> list:
    """Expand demos across seeds, holding the opted-out cells at seed 0.

    Args:
        demo_list (list[Demo]): the entries to bake.
        seeds (list[int]): the seeds asked for.

    Returns:
        list[tuple]: (demo, seed) pairs in demo order, seeds ascending.
    """
    plan = []
    for demo in demo_list:
        want = list(dict.fromkeys(seeds)) if demo.seed_axis else [0]
        plan.extend((demo, seed) for seed in sorted(want))
    return plan


def demo_cell(demo: Demo, seed: int):
    """Return the declared kwargs naming one demo's cell at one seed.

    A runtime_num_vox entry takes its data cell from that cache's grid,
    whose seeds start at the cache's own offset, so seed indexes into it
    and the cell is the one the runtime figure timed.

    Args:
        demo (Demo): the entry.
        seed (int): the data seed, 0-based.

    Returns:
        kwargs_data (dict): one data cell (see benchmark.data.build_clean).
        kwargs_effect (dict | None): one effect cell, None on the null path.

    Raises:
        ValueError: seed is past the runtime cache's seed range.
    """
    kwargs_effect = config.effect_grid(**demo.effect)[0]
    if demo.cache != 'runtime_num_vox':
        kwargs_data = config.data_grid(seeds=[seed], sources=[SOURCE],
                                       **demo.data)[0]
        return kwargs_data, kwargs_effect

    offset = config.RUNTIME_SEED_OFFSET[demo.cache]
    cells = config.runtime_data_grid(
        seed_offset=offset, crop_n_vox_list=[demo.data['crop_n_vox']])
    hit = [kwargs for kwargs in cells if kwargs['seed'] == offset + seed]
    if not hit:
        raise ValueError(f'{demo.key}: {demo.cache} runs seeds '
                         f'0..{len(cells) - 1}, not {seed}')
    return hit[0], kwargs_effect


def realize_cell(kwargs_data, kwargs_effect):
    """Return one cell's realization without writing to the store.

    A cell already in the store is read back, so its plant is the one the
    figures scored. Any other runs get_exp_effect's raw function, unwrapped
    past its cache and recorder: a fresh call would file a provenance
    record for a cell no cache declared.

    Args:
        kwargs_data (dict): one data cell.
        kwargs_effect (dict | None): one effect cell, or None.

    Returns:
        cell (ExpEffect): the realized cell.
    """
    if get_exp_effect.check_call_in_cache(kwargs_data, kwargs_effect):
        return get_exp_effect(kwargs_data, kwargs_effect)
    return inspect.unwrap(get_exp_effect)(kwargs_data, kwargs_effect)


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


def build_demo(demo: Demo, *, seed: int = 0, verbose: bool = True):
    """Build one demo's cell and fit the paper's recipe on it.

    The fit is benchmark.run.run_ana's: the rebuilt raw experiment, with
    the analysis scaling it on the way in.

    Args:
        demo (Demo): the entry to build.
        seed (int): the data seed. The effect placement seeds from the
            data cell, so this redraws the plant along with the design
            (and, for HCP, the feature subset), which is the paper's own
            notion of a random seed.
        verbose (bool): forward progress to the analysis fit.

    Returns:
        ana (AnalysisGLOW): the fitted analysis.
        exp (Experiment): the experiment it was fit on, effect included,
            with no source attached.
        mask_target (np.array | None): the realized (X, Y, Z) bool
            support, or None on the null path.
    """
    cell = realize_cell(*demo_cell(demo, seed))
    exp = cell.build()
    print(f'    y={exp.y.shape} x={exp.x.shape}')

    mask_target = None
    if cell.effect_list:
        mask_target = cell.mask_target_list[0]
        print(f'    planted {int(mask_target.sum())} voxels')

    ana = paper_ana(**demo.ana)
    fit_params = {**config.GLOW_FIT_PARAMS, **demo.fit}
    print(f'    fitting {ana!r}')
    ana.fit(exp, verbose=verbose, **fit_params)
    print(f'    {len(ana.effect_list)} effect(s) discovered')

    # the source's class lives in glow._extra.benchmark, which the image
    # leaves out, so a bundle carrying it would not unpickle there; the
    # viewer never reads images back.
    exp.source = None
    return ana, exp, mask_target


def demo_params(demo: Demo, seed: int = 0) -> dict:
    """Describe one demo by the parameter values that name its cell.

    Reads the same grids build_demo does, so a value is the cell's own
    rather than a restatement of the override that produced it -- a demo
    that overrides nothing still reports the axis default it inherited,
    which is what a picker has to offer. Builds nothing: a grid cell is
    kwargs until a factory is called on it.

    Args:
        demo (Demo): the entry to describe.
        seed (int): the data seed the bundle was built at.

    Returns:
        dict: {source, seed, b, num_img, num_vox, effect_llr, n_vox_frac,
            cluster_mode, prune_rule, keep_stat, alpha_fwer,
            n_perm_fwer, n_perm_inner}. effect_llr and n_vox_frac are
            None on the null path, where nothing is planted.
    """
    kwargs_data, kwargs_effect = demo_cell(demo, seed)
    ana = paper_ana(**demo.ana)
    planted = kwargs_effect or {}
    # an HCP cell names its features instead of carrying b, and draws the
    # whole cohort rather than a chosen num_img
    return dict(
        source=SOURCE,
        seed=int(seed),
        b=len(kwargs_data['hcp_feats']),
        num_img=int(config.HCP_NUM_IMG),
        num_vox=int(kwargs_data['extenter'].n_vox),
        effect_llr=(float(planted['effect_llr']) if planted else None),
        n_vox_frac=(float(planted['n_vox_frac']) if planted else None),
        cluster_mode=ana.cluster_mode.name,
        prune_rule=str(ana.prune_rule),
        keep_stat=bool(getattr(ana, 'keep_stat', False)),
        alpha_fwer=float(ana.alpha_fwer),
        n_perm_fwer=int(ana.n_perm_fwer),
        n_perm_inner=int(ana.n_perm_inner))


def demo_stats(ana, mask_target, mask_active) -> dict:
    """Score one baked fit the way the paper's figures score it.

    Runs the catalogue's own score_effects and reduces it to what a
    reader can hold at once: the overlap trio from
    glow.mask.stats_from_counts and the structural pair from plot's
    split-VI (Rosenberg & Hirschberg 2007). Both come from the figures'
    own definitions, so a demo's numbers cannot drift from the figure's.

    Args:
        ana: the fitted analysis.
        mask_target (np.array | None): (X, Y, Z) bool planted support,
            None on the null path.
        mask_active (np.array): (X, Y, Z) bool, the analyzed voxels.

    Returns:
        dict: {n_pred, min_pval, dice, sens, ppv, hom, com}. The five
            scores are absent on the null path, where an overlap with an
            empty target would report 0 for having nothing to find. A
            score that is undefined rather than absent is None.
    """
    target_list = [] if mask_target is None else [mask_target]
    score = score_effects(ana, target_list, mask_active)

    def clean(x):
        """Return a json-safe float, NaN becoming None."""
        return None if x is None or not np.isfinite(x) else float(x)

    out = {'n_pred': int(score['n_pred']),
           'min_pval': clean(score['min_pval'])}
    if not target_list:
        return out

    counts = {k: np.array([score['target'][k]])
              for k in ('tp', 'fp', 'tn', 'fn')}
    overlap = glow.mask.stats_from_counts(**counts)
    out.update({k: clean(overlap[k][0]) for k in ('dice', 'sens', 'ppv')})

    n_r = np.array([[r['num_vox'] for r in score['pred']]], dtype=float)
    o_r = np.array([[r['target'] for r in score['pred']]], dtype=float)
    hom, com = _hom_com(n_r.reshape(1, -1), o_r.reshape(1, -1))
    out['hom'] = clean(hom[0])
    out['com'] = clean(com[0])
    return out


def score_bundle(path: pathlib.Path) -> dict:
    """Unpickle one baked bundle and score it.

    Args:
        path (pathlib.Path): the .p.gz bundle.

    Returns:
        dict: demo_stats for the fit it holds.
    """
    with gzip.open(path, 'rb') as f:
        payload = pickle.load(f)
    return demo_stats(payload['ana'], payload.get('mask_target'),
                      payload['exp'].mask_idx > -1)


def write_manifest(out_dir: pathlib.Path) -> None:
    """Write manifest.json describing every bundle present in out_dir.

    Reads the directory rather than the bake plan, so the manifest
    describes whatever is on disk however it got there -- a later run
    adding one seed does not drop the seeds already baked. The server
    reads it at boot so the landing page can name the set and build its
    picker without unpickling it. Each entry carries demo_params, which
    the dropdowns are built from, and demo_stats, which the picker shows
    under them.

    Scoring is the one part that has to open a bundle, so the manifest
    already on disk is reused as its own cache, keyed by name and size:
    only a bundle that is new or refit is unpickled. Delete the manifest
    to force a full rescore.

    Args:
        out_dir (pathlib.Path): the bundle directory.
    """
    prior = {}
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists():
        for old in json.loads(manifest_path.read_text()).get('demos', []):
            if 'stats' in old:
                prior[(old['key'], old.get('size_bytes'))] = old['stats']
    order = {d.key: i for i, d in enumerate(DEMOS)}
    found = []
    for path in out_dir.glob('*.p.gz'):
        named = parse_bundle_name(path.name.removesuffix('.p.gz'))
        if named is not None:
            found.append((order[named[0].key], named[1], path, *named))

    entries = []
    for _, seed, path, demo, _ in sorted(found, key=lambda r: r[:2]):
        key = path.name.removesuffix('.p.gz')
        size = path.stat().st_size
        blurb = demo.blurb if seed == 0 else f'{demo.blurb} -- seed {seed}'
        stats = prior.get((key, size))
        if stats is None:
            print(f'  scoring {key} ...')
            sys.stdout.flush()
            stats = score_bundle(path)
        entries.append({'key': key,
                        'blurb': blurb,
                        'cache': demo.cache,
                        'caches': [demo.cache, *demo.also],
                        'source': SOURCE,
                        'size_bytes': size,
                        'params': demo_params(demo, seed),
                        'stats': stats})
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
        '--seeds', type=int, nargs='+', default=[0], metavar='N',
        help='data seeds to bake (default 0). A cell whose seed_axis is '
             'off ignores this and stays at seed 0.')
    parser.add_argument(
        '--list', action='store_true',
        help='print the plan and exit without building')
    args = parser.parse_args()

    demo_list = DEMOS
    if args.only:
        wanted = set(args.only)
        unknown = wanted - set(_DEMO_BY_KEY)
        if unknown:
            parser.error(f'unknown --only key(s): {sorted(unknown)}')
        demo_list = [d for d in DEMOS if d.key in wanted]

    plan = bake_plan(demo_list, args.seeds)

    if args.list:
        for demo, seed in plan:
            print(f'{demo_key(demo, seed):<26} {demo.cache:<18} '
                  f'{demo.blurb}')
        print(f'{len(plan)} entr(ies)')
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    print(f'baking {len(plan)} demo(s) into {args.out}')

    total_t0 = time.time()
    for i, (demo, seed) in enumerate(plan, 1):
        key = demo_key(demo, seed)
        out = args.out / f'{key}.p.gz'
        head = f'[{i}/{len(plan)}] {key}'

        if out.exists() and not args.force:
            size_mb = out.stat().st_size / (1024 ** 2)
            print(f'{head}: skip (exists, {size_mb:.1f} MB)')
            continue

        print(f'{head}: {demo.blurb}')
        t0 = time.time()
        ana, exp, mask_target = build_demo(demo, seed=seed)
        build_s = time.time() - t0

        with gzip.open(out, 'wb') as f:
            pickle.dump({'ana': ana, 'exp': exp, 'mask_target': mask_target,
                         'demo': dataclasses.asdict(demo), 'seed': seed}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = out.stat().st_size / (1024 ** 2)
        print(f'    -> {out.name} ({size_mb:.1f} MB, {build_s:.0f}s)')
        sys.stdout.flush()

    write_manifest(args.out)
    print(f'done in {time.time() - total_t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
