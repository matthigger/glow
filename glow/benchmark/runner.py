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

_RUNTIME_ANA_KEYS = {'perm_dir', 'n_jobs_perm'}


def ana_entry(cls, kw):
    """Canonical, hashable representation of an analysis kwargs tuple."""
    entry = {'class': cls.__name__}
    for k, v in sorted(kw.items()):
        if k in _RUNTIME_ANA_KEYS:
            continue
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
         'stat': ana.get_stat.__name__.replace('get_', ''),
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

    if config.detail_save:
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
        """Yield (label, ana_kwargs) for worker-side mutation (n_jobs_perm=1).

        Default: no analyses.  Overridden by subclasses that hold
        ana_kwargs dicts that need runtime knob patching.
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
        exp, effect = config.get_exp_eff(**iter_kw)

        for ana_label, (Ana, ana_kw) in self.ana_kwargs_dict.items():
            if _skip_labels and ana_label in _skip_labels:
                continue
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
                    continue
            else:
                ana = Ana(exp=exp, **ana_kw)
            total_time_sec = time.time() - start

            _score_and_emit(ana, effect, config, ana_label,
                            total_time_sec, iter_kw)


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
    """Run one AnalysisGLOW then apply six pruning methods.

    AnalysisGLOW is hard-coded since the prune variants all assume it
    (they consume its children, sig_reg_list, llr_adjusted_0, etc.).
    """

    PRUNE_LABELS = ('greedy', 'greedy_adj', 'dp_lam0', 'dp_lam0_adj',
                    'dp_geom3', 'full_adjust')

    def __init__(self, glow_ana_kwargs):
        self.glow_ana_kwargs = glow_ana_kwargs

    @property
    def labels(self):
        return set(self.PRUNE_LABELS)

    def label_recipe(self, label):
        # label (the prune-method string) goes in via the base recipe
        return {'ana': ana_entry(glow.analysis.AnalysisGLOW,
                                 self.glow_ana_kwargs)}

    def iter_ana_kwargs(self):
        yield 'GLOW', (glow.analysis.AnalysisGLOW, self.glow_ana_kwargs)

    def run(self, config, **iter_kw):
        from glow.analysis.prune import (prune_greedy, prune_dp,
                                         prune_greedy_full_adjust)

        exp, effect = config.get_exp_eff(**iter_kw)

        start = time.time()
        ana = glow.analysis.AnalysisGLOW(exp=exp, **self.glow_ana_kwargs)
        total_time_sec = time.time() - start

        sig = ana.sig_reg_list
        children = ana.children
        stat = np.nan_to_num(ana.stat.ravel().astype(float),
                             nan=0.0, posinf=0.0, neginf=0.0)
        stat_adj = np.nan_to_num(ana.llr_adjusted_0.ravel().astype(float),
                                 nan=0.0, posinf=0.0, neginf=0.0)

        methods = {
            'greedy': prune_greedy(sig, children, stat),
            'greedy_adj': prune_greedy(sig, children, stat_adj),
            'dp_lam0': prune_dp(sig, children, stat, lam=0.0),
            'dp_lam0_adj': prune_dp(sig, children, stat_adj, lam=0.0),
            'dp_geom3': prune_dp(sig, children, stat, exp_n_eff=3.0),
            'full_adjust': prune_greedy_full_adjust(sig, children, exp),
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
                 'stat': ana.get_stat.__name__.replace('get_', ''),
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
# run-mancova-glow: all MANCOVA stats sharing one E/H walk
# ---------------------------------------------------------------------------

class RunMancovaGlow(Runner):
    """Run GLOW with all MANCOVA stats, sharing E/H across stats."""

    def __init__(self, glow_ana_kwargs):
        self.glow_ana_kwargs = glow_ana_kwargs

    @property
    def labels(self):
        from glow.analysis.mancova import stat_dict
        return {f'GLOW-{name}' for name in stat_dict}

    def label_recipe(self, label):
        stat_name = label.split('-', 1)[1]
        return {'ana': ana_entry(glow.analysis.AnalysisGLOW,
                                 self.glow_ana_kwargs),
                'stat': stat_name}

    def iter_ana_kwargs(self):
        yield 'GLOW', (glow.analysis.AnalysisGLOW, self.glow_ana_kwargs)

    def run(self, config, **iter_kw):
        from glow.analysis import (
            Analysis, AnalysisGLOW, _sanitize_adjusted_stat)
        from glow.analysis.cluster import cluster
        from glow.analysis.mancova import (
            stat_dict, stat_dict_inv, get_llr)

        exp, effect = config.get_exp_eff(**iter_kw)
        start = time.time()

        stat_fns = list(stat_dict.values())
        ana_kw = self.glow_ana_kwargs
        n_perm_fwer = ana_kw['n_perm_fwer']
        n_perm_sa = ana_kw.get('n_perm_fwer_size_adjust', 50)
        alpha_fwer = ana_kw.get('alpha_fwer', 0.05)
        min_size = ana_kw.get('min_size', 1)

        fit_start = n_perm_fwer + 1
        fit_end = n_perm_fwer + n_perm_sa
        num_vox = exp.y.shape[2]

        fit_sizes_all = {fn: [] for fn in stat_fns}
        fit_stats_all = {fn: [] for fn in stat_fns}

        for perm_idx in range(fit_start, fit_end + 1):
            _exp = exp.permute(perm_idx)
            children = cluster(exp=_exp)
            multi = Analysis.get_stat_perm_multi(
                exp=_exp, get_stat_list=stat_fns, children=children)
            size = glow.graph.node_sum(
                np.ones(num_vox, dtype=int), children)
            for fn in stat_fns:
                fit_sizes_all[fn].append(size.astype(float))
                fit_stats_all[fn].append(multi[fn].ravel().astype(float))

        mu_fns = {}
        gams = {}
        for fn in stat_fns:
            sz = np.concatenate(fit_sizes_all[fn])
            st = np.concatenate(fit_stats_all[fn])
            gam, mu_fn, _r2 = AnalysisGLOW.fit_size_gam(sz, st)
            mu_fns[fn] = mu_fn
            gams[fn] = gam

        stat_max = {fn: [] for fn in stat_fns}
        children_0 = size_0 = stat_0 = None

        for perm_idx in range(n_perm_fwer + 1):
            _exp = exp.permute(perm_idx)
            children = cluster(exp=_exp)
            multi = Analysis.get_stat_perm_multi(
                exp=_exp, get_stat_list=stat_fns, children=children)
            size = glow.graph.node_sum(
                np.ones(num_vox, dtype=int), children).astype(float)

            active = size >= min_size
            for fn in stat_fns:
                adj = multi[fn].ravel() - mu_fns[fn](size)
                adj = _sanitize_adjusted_stat(adj)
                stat_max[fn].append(
                    float(np.nanmax(adj[active])) if active.any()
                    else float('-inf'))

            if perm_idx == 0:
                children_0 = children
                size_0 = size
                stat_0 = {fn: multi[fn].ravel().astype(float)
                          for fn in stat_fns}

        stat_max_sorted = {fn: np.sort(stat_max[fn]) for fn in stat_fns}
        total_time = time.time() - start

        llr_prune = stat_0[get_llr]

        for fn in stat_fns:
            name = stat_dict_inv[fn]
            ana = AnalysisGLOW.from_precomputed(
                exp=exp, get_stat=fn, adj_gam=gams[fn])
            ana._finalize_analysis(
                exp, n_perm_fwer,
                stat_0[fn], size_0, children_0,
                mu_fns[fn], stat_max_sorted[fn],
                alpha_fwer, min_size,
                prune_stat=llr_prune)

            _score_and_emit(ana, effect, config,
                            f'GLOW-{name}', total_time, iter_kw)


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
            Analysis, AnalysisVBA, AnalysisCET, DEFAULT_CET_CFT_PVAL)
        from glow.analysis.mancova import stat_dict, stat_dict_inv

        exp, effect = config.get_exp_eff(**iter_kw)
        start = time.time()

        stat_fns = list(stat_dict.values())
        ana_kw = self.vba_ana_kwargs
        n_perm_fwer = ana_kw['n_perm_fwer']
        alpha_fwer = ana_kw.get('alpha_fwer', 0.05)

        multi = Analysis.get_stat_perm_multi(
            exp, stat_fns, n_perm=n_perm_fwer, children=None)
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
