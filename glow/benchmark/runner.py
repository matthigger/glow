"""Runner classes: encapsulate what one experiment produces and how it's hashed.

A Runner owns (a) the emission of result rows for one (seed, effect_llr)
experiment and (b) the per-row hash that identifies its "recipe" --- the
inputs that would produce the same row.  Separating this from Config
prevents the landmine where the set of labels a run_fnc emits and the
labels used to compute the config_hash could silently disagree.
"""
from abc import ABC, abstractmethod
import gzip
import hashlib
import json
import time
import traceback

import cloudpickle as pickle
import numpy as np

import glow
from glow.benchmark.file import OUT, ERROR, short_uuid


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _stat_name(ana):
    """Stat-function label for a result row.

    AnalysisGLOW is LLR-only and has no ``get_stat`` attribute; the
    voxel-stat analyses (VBA, CET) expose the pluggable function.
    """
    fn = getattr(ana, 'get_stat', None)
    if fn is None:
        return 'llr'
    return fn.__name__.replace('get_', '')


def ana_entry(cls, kw):
    """Canonical, hashable representation of an analysis kwargs tuple."""
    entry = {'class': cls.__name__}
    for k, v in sorted(kw.items()):
        entry[k] = v.__name__ if callable(v) else v
    return entry


def _jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _merge_iter_kw(d, iter_kw):
    for k, v in iter_kw.items():
        if k not in d:
            d[k] = _jsonable(v)


def _write_result(config, d, *, subfolder=OUT):
    uuid_str = short_uuid()
    d['uuid'] = uuid_str
    file_out = config.folder / subfolder / f'{uuid_str}_result.json'
    file_out.parent.mkdir(exist_ok=True, parents=True)
    with open(file_out, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)
    return uuid_str


def _score_and_emit(ana, effect, config, label, total_time_sec, iter_kw):
    exp = ana.exp

    mask_pred = np.zeros(exp.mask_idx.shape, dtype=bool)
    for _effect in ana.effect_list:
        mask_pred |= _effect.mask

    mask_active = exp.mask_idx > -1
    dice, sens, spec = glow.mask.get_score(mask_pred=mask_pred,
                                           mask_target=effect.mask,
                                           mask_active=mask_active)

    pct_max_dice = 0.0
    if hasattr(ana, 'sig_reg_list') and hasattr(ana, 'children'):
        dice_all, _, _ = glow.graph.get_dice_sens_spec(
            mask=effect.mask, mask_idx=exp.mask_idx,
            children=ana.children)
        sig = ana.sig_reg_list
        max_dice_sig = max((dice_all[i] for i in sig), default=0.0)
        if max_dice_sig > 0:
            out_regs = [eff.reg_idx for eff in ana.effect_list]
            if out_regs:
                pct_max_dice = float(
                    max(dice_all[i] for i in out_regs) / max_dice_sig)

    min_pval = float(np.nanmin(ana.pval)) if hasattr(ana, 'pval') else None

    d = {'effect_llr': effect.effect_llr,
         'seed': int(effect.seed),
         'stat': _stat_name(ana),
         'label': label,
         'Analysis': type(ana).__name__,
         'dice': dice,
         'sens': sens,
         'spec': spec,
         'pct_max_dice': pct_max_dice,
         'min_pval': min_pval,
         'vox_total': int(exp.y.shape[2]),
         'vox_effect': int(effect.mask.sum()),
         'time_sec': total_time_sec,
         'config_hash': config.runner.hash(config, label)}
    _merge_iter_kw(d, iter_kw)
    uuid_str = _write_result(config, d)

    # Always save the analysis pickle.  Slim pickling drops y while
    # preserving the recipe; ~100 KB per (ana, effect) is well within
    # any reasonable per-experiment budget.
    file_out = config.folder / OUT / f'{uuid_str}_detail.p.gz'
    with gzip.open(file_out, 'wb') as f:
        pickle.dump((ana, effect), f)


# ---------------------------------------------------------------------------
# Runner base
# ---------------------------------------------------------------------------

class Runner(ABC):
    """ABC for runners.  Subclasses own result emission + per-label hashing."""

    @property
    @abstractmethod
    def labels(self):
        """set of label strings this runner emits per experiment."""

    @abstractmethod
    def run(self, config, **iter_kw):
        """run one experiment, writing JSON results for each label."""

    def label_recipe(self, label):
        """Per-label contribution to the hash (merged into base_recipe)."""
        return {}

    def recipe(self, config, label):
        return {**config.base_recipe(),
                'label': label,
                **self.label_recipe(label)}

    def hash(self, config, label):
        sig = json.dumps(self.recipe(config, label),
                         sort_keys=True, default=str)
        return hashlib.sha256(sig.encode()).hexdigest()[:12]

    def iter_ana_kwargs(self):
        """Yield (label, ana_kwargs) entries this runner would run.

        Default: no analyses.  Overridden by subclasses that hold
        ana_kwargs dicts.
        """
        return iter(())

    # Introspection used by runtime/memory estimators -----------------------

    def representative_ana_kw(self):
        """Return (Ana, kw) to use for memory/timeout estimation, or None."""
        for _label, entry in self.iter_ana_kwargs():
            return entry
        return None

    def to_dict(self):
        """Serializable view of this runner (used by Config.save_config)."""
        def _conv(v):
            if isinstance(v, dict):
                return {k: _conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_conv(x) for x in v]
            if isinstance(v, type) or callable(v):
                return getattr(v, '__name__', repr(v))
            return v
        return {'class': type(self).__name__, **_conv(self.__dict__)}


# ---------------------------------------------------------------------------
# run-ana: multiple analyses per experiment
# ---------------------------------------------------------------------------

class RunAna(Runner):
    """Run every (Ana, kwargs) in ana_kwargs_dict; one result row per label."""

    def __init__(self, ana_kwargs_dict):
        self.ana_kwargs_dict = ana_kwargs_dict

    @property
    def labels(self):
        return set(self.ana_kwargs_dict)

    def label_recipe(self, label):
        cls, kw = self.ana_kwargs_dict[label]
        return {'ana': ana_entry(cls, kw)}

    def iter_ana_kwargs(self):
        for label, entry in self.ana_kwargs_dict.items():
            yield label, entry

    def run(self, config, _skip_labels=None, **iter_kw):
        from collections import defaultdict
        from glow.analysis import AnalysisVBA, AnalysisCET

        exp, effect = config.get_exp_eff(**iter_kw)

        # VBA / VBA-TFCE / CET share the per-perm voxel stat walk
        # (the bulk of total cost).  Group by n_perm_fwer, run one
        # multi-stat walk per group, then dispatch each label via
        # Ana.from_precomputed.  Other analyses (e.g. AnalysisGLOW)
        # fall through to per-entry construction.
        shared = defaultdict(list)
        other = []
        for ana_label, (Ana, ana_kw) in self.ana_kwargs_dict.items():
            if _skip_labels and ana_label in _skip_labels:
                continue
            if Ana in (AnalysisVBA, AnalysisCET):
                shared[ana_kw['n_perm_fwer']].append(
                    (ana_label, Ana, ana_kw))
            else:
                other.append((ana_label, Ana, ana_kw))

        for _n_perm, members in shared.items():
            self._run_shared_group(config, exp, effect, members, iter_kw)

        for ana_label, Ana, ana_kw in other:
            self._run_single(config, exp, effect, ana_label, Ana, ana_kw,
                             iter_kw)

    @staticmethod
    def _run_single(config, exp, effect, ana_label, Ana, ana_kw, iter_kw):
        start = time.time()
        if config.error_save:
            try:
                ana = Ana(exp=exp, **ana_kw)
            except Exception:
                d = {'error_msg': traceback.format_exc(),
                     'label': ana_label,
                     'method': Ana.__name__,
                     'effect_llr': effect.effect_llr,
                     'seed': int(effect.seed)}
                print(f'error: {d}')
                _write_result(config, d, subfolder=ERROR)
                return
        else:
            ana = Ana(exp=exp, **ana_kw)
        total_time_sec = time.time() - start
        _score_and_emit(ana, effect, config, ana_label, total_time_sec,
                        iter_kw)

    @staticmethod
    def _run_shared_group(config, exp, effect, members, iter_kw):
        """Run one shared voxel-stat walk and dispatch each member.

        Each member's reported ``time_sec`` is ``walk_time + own_post``
        — the cost it would incur run in isolation — matching the
        existing ``RunMancovaVba`` convention.
        """
        try:
            stat_by_fn, walk_time = _run_shared_voxel_walk(exp, members)
        except Exception:
            if not config.error_save:
                raise
            tb = traceback.format_exc()
            for ana_label, Ana, _kw in members:
                d = {'error_msg': tb,
                     'label': ana_label,
                     'method': Ana.__name__,
                     'effect_llr': effect.effect_llr,
                     'seed': int(effect.seed)}
                print(f'error: {d}')
                _write_result(config, d, subfolder=ERROR)
            return

        for ana_label, Ana, ana_kw in members:
            post_start = time.time()
            if config.error_save:
                try:
                    ana = _dispatch_shared(Ana=Ana, exp=exp, ana_kw=ana_kw,
                                           stat_by_fn=stat_by_fn)
                except Exception:
                    d = {'error_msg': traceback.format_exc(),
                         'label': ana_label,
                         'method': Ana.__name__,
                         'effect_llr': effect.effect_llr,
                         'seed': int(effect.seed)}
                    print(f'error: {d}')
                    _write_result(config, d, subfolder=ERROR)
                    continue
            else:
                ana = _dispatch_shared(Ana=Ana, exp=exp, ana_kw=ana_kw,
                                       stat_by_fn=stat_by_fn)
            total_time_sec = walk_time + (time.time() - post_start)
            _score_and_emit(ana, effect, config, ana_label,
                            total_time_sec, iter_kw)


def _run_shared_voxel_walk(exp, members):
    """Compute one (n_perm+1, num_vox) stat matrix per unique get_stat fn.

    Args:
        exp: Experiment (unscaled or scaled; permute() works on both).
        members: list of (label, Ana, ana_kw).  All entries share the
            same ``n_perm_fwer``; ``get_stat`` may differ — multiple stats
            share one E/H tree walk via ``get_stat_perm_multi``.

    Returns:
        (stat_by_fn, walk_time): dict mapping each stat fn to its
        (n_perm+1, num_vox) matrix, and the wall-clock seconds spent.
    """
    from glow.analysis import AnalysisVoxel
    from glow.analysis.mancova import get_wilks

    n_perm_fwer = members[0][2]['n_perm_fwer']

    stat_fns = []
    for _, _, kw in members:
        fn = kw.get('get_stat') or get_wilks
        if fn not in stat_fns:
            stat_fns.append(fn)

    walk_start = time.time()
    num_vox = exp.y.shape[2]
    stat_by_fn = {fn: np.full((n_perm_fwer + 1, num_vox), np.nan)
                  for fn in stat_fns}
    for k in range(n_perm_fwer + 1):
        _exp = exp.permute(k) if k else exp
        row = AnalysisVoxel.get_stat_perm_multi(_exp, stat_fns, children=None)
        for fn in stat_fns:
            stat_by_fn[fn][k, :] = row[fn]
    return stat_by_fn, time.time() - walk_start


def _dispatch_shared(*, Ana, exp, ana_kw, stat_by_fn):
    """Apply per-label post-processing and call Ana.from_precomputed.

    z_flag is honored before TFCE (matching AnalysisVBA.__init__) and
    before CFT computation (matching AnalysisCET.__init__).
    """
    from glow.analysis import (
        Analysis, AnalysisVBA, AnalysisCET, DEFAULT_CET_CFT_PVAL)
    from glow.analysis.mancova import get_wilks

    get_stat = ana_kw.get('get_stat') or get_wilks
    alpha_fwer = ana_kw.get('alpha_fwer', 0.05)
    z_flag = ana_kw.get('z_flag', False)

    stat = stat_by_fn[get_stat].copy()
    if z_flag:
        stat = Analysis.z_score_stat(stat)

    if Ana is AnalysisVBA:
        if ana_kw.get('tfce_flag', False):
            stat = AnalysisVBA.apply_tfce(
                stat=stat, mask_idx=exp.mask_idx,
                verbose=ana_kw.get('verbose', False))
        return AnalysisVBA.from_precomputed(
            exp=exp, get_stat=get_stat, stat=stat, alpha_fwer=alpha_fwer)

    if Ana is AnalysisCET:
        cft_pval = ana_kw.get('cft_pval', DEFAULT_CET_CFT_PVAL)
        null_pool = stat[1:, :].ravel()
        cft = np.quantile(null_pool, 1 - cft_pval)
        return AnalysisCET.from_precomputed(
            exp=exp, get_stat=get_stat, stat=stat, cft=cft,
            cft_pval=cft_pval, alpha_fwer=alpha_fwer, z_flag=z_flag)

    raise AssertionError(f'unexpected Ana class: {Ana}')


# ---------------------------------------------------------------------------
# run-segment: Ward's clustering, no analyses
# ---------------------------------------------------------------------------

class RunSegment(Runner):
    """Run Ward's clustering variants and score against the imposed effect."""

    @property
    def labels(self):
        from glow.analysis.cluster import MODE_LABELS
        return set(MODE_LABELS.values())

    def run(self, config, **iter_kw):
        from glow.analysis.cluster import _MODES, MODE_LABELS, cluster

        exp, effect = config.get_exp_eff(**iter_kw)

        for mode in _MODES:
            start = time.time()
            children = cluster(exp, mode=mode)
            total_time_sec = time.time() - start

            dice, sens, spec = glow.graph.get_dice_sens_spec(
                mask=effect.mask,
                mask_idx=exp.mask_idx,
                children=children)
            idx = np.argmax(dice)

            label = MODE_LABELS[mode]
            d = {'effect_llr': effect.effect_llr,
                 'seed': int(effect.seed),
                 'dice': dice[idx],
                 'label': label,
                 'sens': sens[idx],
                 'spec': spec[idx],
                 'vox_total': int(exp.y.shape[2]),
                 'vox_effect': int(effect.mask.sum()),
                 'time_sec': total_time_sec,
                 'config_hash': config.runner.hash(config, label)}
            _merge_iter_kw(d, iter_kw)
            _write_result(config, d)


# ---------------------------------------------------------------------------
# run-prune-compare: six pruning methods on top of AnalysisGLOW
# ---------------------------------------------------------------------------

class RunPruneCompare(Runner):
    """Run one AnalysisGLOW then prune by raw LLR vs per-region z-LLR.

    Compares two pruning gain choices on the same significant-region
    set: raw LLR (``stat``) vs the per-region z-scored LLR
    (``llr_z_0``).  Both greedy and DP-antichain are tried,
    giving four labels.
    """

    PRUNE_LABELS = ('greedy_llr', 'greedy_z', 'dp_llr', 'dp_z')

    def __init__(self, glow_ana_kwargs):
        self.glow_ana_kwargs = glow_ana_kwargs

    @property
    def labels(self):
        return set(self.PRUNE_LABELS)

    def label_recipe(self, label):
        return {'ana': ana_entry(glow.analysis.AnalysisGLOW,
                                 self.glow_ana_kwargs)}

    def iter_ana_kwargs(self):
        yield 'GLOW', (glow.analysis.AnalysisGLOW, self.glow_ana_kwargs)

    def run(self, config, **iter_kw):
        from glow.analysis.prune import prune_greedy, prune_dp

        exp, effect = config.get_exp_eff(**iter_kw)

        start = time.time()
        ana = glow.analysis.AnalysisGLOW(exp=exp, **self.glow_ana_kwargs)
        total_time_sec = time.time() - start

        sig = ana.sig_reg_list
        children = ana.children
        llr = np.nan_to_num(ana.stat.astype(float),
                            nan=0.0, posinf=0.0, neginf=0.0)
        llr_z = np.nan_to_num(ana.llr_z_0.astype(float),
                              nan=0.0, posinf=0.0, neginf=0.0)

        methods = {
            'greedy_llr': prune_greedy(sig, children, llr),
            'greedy_z':   prune_greedy(sig, children, llr_z),
            'dp_llr':     prune_dp(sig, children, llr, lam=0.0),
            'dp_z':       prune_dp(sig, children, llr_z, lam=0.0),
        }

        mask_active = exp.mask_idx > -1

        for prune_label, (reg_out_list, _info) in methods.items():
            mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
            for reg_idx in reg_out_list:
                label_map = glow.graph.get_label_map(
                    reg_idx_list=[reg_idx],
                    mask_idx=exp.mask_idx,
                    children=children)
                mask_pred |= (label_map > -1)

            dice, sens, spec = glow.mask.get_score(mask_pred=mask_pred,
                                                   mask_target=effect.mask,
                                                   mask_active=mask_active)
            d = {'effect_llr': effect.effect_llr,
                 'seed': int(effect.seed),
                 'stat': _stat_name(ana),
                 'label': prune_label,
                 'Analysis': type(ana).__name__,
                 'dice': dice,
                 'sens': sens,
                 'spec': spec,
                 'vox_total': int(ana.exp.y.shape[2]),
                 'vox_effect': int(effect.mask.sum()),
                 'n_sig': len(sig),
                 'n_selected': len(reg_out_list),
                 'time_sec': total_time_sec,
                 'config_hash': config.runner.hash(config, prune_label)}
            _merge_iter_kw(d, iter_kw)
            _write_result(config, d)


# ---------------------------------------------------------------------------
# run-mancova-vba: VBA / VBA-TFCE / CET x 5 stats x {raw, z}
# ---------------------------------------------------------------------------

class RunMancovaVba(Runner):
    """Run VBA, VBA-TFCE, and CET with all 5 MANCOVA stats x {raw, z}."""

    FAMILIES = ('VBA', 'VBA-TFCE', 'CET')

    def __init__(self, vba_ana_kwargs):
        self.vba_ana_kwargs = vba_ana_kwargs
        self._specs = {}
        from glow.analysis.mancova import stat_dict
        for family in self.FAMILIES:
            for stat_name in stat_dict:
                for z_flag in (False, True):
                    suffix = '-z' if z_flag else ''
                    label = f'{family}-{stat_name}{suffix}'
                    self._specs[label] = {
                        'family': family,
                        'stat': stat_name,
                        'z_flag': z_flag,
                    }

    @property
    def labels(self):
        return set(self._specs)

    def label_recipe(self, label):
        # The VBA/CET distinction drives which AnalysisCls is used at runtime;
        # families share the vba_ana_kwargs at the config level.
        return {'ana': ana_entry(glow.analysis.AnalysisVBA,
                                 self.vba_ana_kwargs),
                **self._specs[label]}

    def iter_ana_kwargs(self):
        yield 'VBA', (glow.analysis.AnalysisVBA, self.vba_ana_kwargs)

    def run(self, config, **iter_kw):
        from glow.analysis import (
            Analysis, AnalysisVoxel, AnalysisVBA, AnalysisCET,
            DEFAULT_CET_CFT_PVAL)
        from glow.analysis.mancova import stat_dict, stat_dict_inv

        exp, effect = config.get_exp_eff(**iter_kw)
        start = time.time()

        stat_fns = list(stat_dict.values())
        ana_kw = self.vba_ana_kwargs
        n_perm_fwer = ana_kw['n_perm_fwer']
        alpha_fwer = ana_kw.get('alpha_fwer', 0.05)

        # row 0 = observed; rows 1..n_perm_fwer = FL nulls
        num_vox = exp.y.shape[2]
        multi = {fn: np.full((n_perm_fwer + 1, num_vox), np.nan)
                 for fn in stat_fns}
        for k in range(n_perm_fwer + 1):
            _exp = exp.permute(k) if k else exp
            row = AnalysisVoxel.get_stat_perm_multi(
                _exp, stat_fns, children=None)
            for fn in stat_fns:
                multi[fn][k, :] = row[fn]
        walk_time = time.time() - start

        for fn in stat_fns:
            name = stat_dict_inv[fn]

            for z_flag in (False, True):
                variant_start = time.time()
                stat = multi[fn].copy()
                if z_flag:
                    stat = Analysis.z_score_stat(stat)
                suffix = '-z' if z_flag else ''

                self._emit_variant(
                    config, effect, exp, fn, f'VBA-{name}{suffix}',
                    stat, alpha_fwer, walk_time, iter_kw, variant_start,
                    AnalysisVBA)

                stat_tfce = AnalysisVBA.apply_tfce(
                    stat=stat, mask_idx=exp.mask_idx)
                self._emit_variant(
                    config, effect, exp, fn, f'VBA-TFCE-{name}{suffix}',
                    stat_tfce, alpha_fwer, walk_time, iter_kw, variant_start,
                    AnalysisVBA)

        cft_pval = DEFAULT_CET_CFT_PVAL
        for fn in stat_fns:
            name = stat_dict_inv[fn]

            for z_flag in (False, True):
                variant_start = time.time()
                stat = multi[fn].copy()
                if z_flag:
                    stat = Analysis.z_score_stat(stat)
                suffix = '-z' if z_flag else ''

                null_pool = stat[1:, :].ravel()
                cft = np.quantile(null_pool, 1 - cft_pval)

                self._emit_variant(
                    config, effect, exp, fn, f'CET-{name}{suffix}',
                    stat, alpha_fwer, walk_time, iter_kw, variant_start,
                    AnalysisCET, cft=cft, cft_pval=cft_pval, z_flag=z_flag)

    @staticmethod
    def _emit_variant(config, effect, exp, fn, label, stat, alpha_fwer,
                      walk_time, iter_kw, variant_start, AnalysisCls,
                      **factory_kw):
        ana = AnalysisCls.from_precomputed(
            exp=exp, get_stat=fn, stat=stat, alpha_fwer=alpha_fwer,
            **factory_kw)
        _score_and_emit(ana, effect, config, label,
                        walk_time + (time.time() - variant_start), iter_kw)
