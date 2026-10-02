"""Re-run a random sample of the paper's leaves and diff them against records.

The deposit (paper_records) claims each leaf is what its declared recipe
computes. This checks that claim the expensive way: it maps every deposit leaf
back to its CONFIG declaration by uid (results.cell_leaf_uids), draws a seeded
random order, and recomputes as many leaves as fit in a wall-clock budget --
the cell realized from scratch, the leaf refit -- then diffs each output
against the record. Paper caches take turns, so a short budget still touches
every figure.

Nothing is read from or written to the benchmark store: cell and leaf run
unwrapped (recipe.raw_fnc), and any call reaching joblib.Memory or the
recorder raises. A leaf's shared heavy intermediate (the prune fit, the stat
walk) is recomputed once and its siblings ride on it, since a later draw would
pay for it again.

Status per leaf: identical (bit for bit), close (ints and strings equal,
floats within --rtol) or differs. Off the machine that produced the records,
expect close at best (rerun.md, 'Why it is not bitwise reproducible'). A
timing leaf is judged on num_vox; its time ratio is logged, not judged.

    python -m glow._extra.benchmark.scripts.validate_records \\
        --seed 0 --runtime 2h --records glow-paper-records.zip

Exits 1 if any leaf differs or fails to run.
"""

import argparse
import collections
import contextlib
import dataclasses
import datetime
import json
import pathlib
import random
import re
import sys
import time
import traceback
import zipfile
from typing import Callable
from unittest import mock

import joblib
import numpy as np

from glow._extra.benchmark import config, results, run
from glow._extra.benchmark.cell import get_exp_effect
from glow._extra.benchmark.file import get_path_records, get_path_result
from glow._extra.benchmark.recipe import raw_fnc
from glow._extra.benchmark.recorder import _cell
from glow._extra.benchmark.scripts.paper_records import (PAPER_CACHES,
                                                          kwargs_data_b)
from glow._extra.benchmark.store import RECORDER

# leaves measuring wall clock: output judged, time only reported
TIME_FNC = ('run_ana_time', 'run_ana_time_1perm')


@dataclasses.dataclass
class Job:
    """One deposit leaf, paired with the declaration that recomputes it.

    Attributes:
        cache (str): the paper cache it is drawn under (first that claims it)
        kwargs_data (dict): declared data cell
        kwargs_effect (dict | None): declared effect cell, None on null path
        kwargs_fnc (dict): declared leaf kwargs
        fnc (Callable): the memoised leaf (unwrapped before it is called)
        key (str): leaf record key in the deposit
        cell_key (str): its cell's get_exp_effect record key
        cell_uid (str): the cell's declared uid (the leaf's parent_uid)
        shared (tuple | None): identity of the shared intermediate this leaf
            reads, None when it has none
    """

    cache: str
    kwargs_data: dict
    kwargs_effect: dict
    kwargs_fnc: dict
    fnc: Callable
    key: str
    cell_key: str
    cell_uid: str
    shared: tuple


def parse_runtime(text: str) -> float:
    """Return seconds from '5400', '90s', '30m', '2h' or '1h30m'."""
    if re.fullmatch(r'\d+(\.\d+)?', text):
        return float(text)
    parts = re.findall(r'(\d+(?:\.\d+)?)([hms])', text)
    if not parts or ''.join(n + u for n, u in parts) != text:
        raise argparse.ArgumentTypeError(f'bad runtime: {text!r}')
    unit = {'h': 3600, 'm': 60, 's': 1}
    return sum(float(n) * unit[u] for n, u in parts)


def load_records(path: pathlib.Path) -> dict:
    """Read a records tree: a records/ dir, its parent, or the deposit zip.

    Args:
        path (pathlib.Path): directory of <hash>.json, a directory holding
            records/, or a zip with records/*.json (figure_csv/ ignored)

    Returns:
        records (dict): record key -> record
    """
    records = {}
    if path.is_file():
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if re.fullmatch(r'(.*/)?records/[^/]+\.json', name):
                    rec = json.loads(zf.read(name))
                    records[rec['hash']] = rec
        return records
    if (path / 'records').is_dir():
        path = path / 'records'
    for p in path.glob('*.json'):
        rec = json.loads(p.read_text())
        records[rec['hash']] = rec
    return records


def _shared_key(fnc_name: str, kwargs_fnc: dict):
    """Return what names a leaf's shared intermediate, or None.

    Mirrors the memo keys in run: glow_fit_for_prune is every prune knob but
    the rule; voxel_stat_walk is the walk's perm count and kernel.
    """
    if fnc_name == 'run_prune':
        return tuple(sorted((k, repr(v)) for k, v in kwargs_fnc.items()
                            if k not in ('rule', 'fit_params')))
    if fnc_name == 'run_stat':
        ana = kwargs_fnc['ana']
        return (ana.n_perm_fwer, ana.fwhm)
    return None


def index_jobs(records: dict, cache_list, source_list=None) -> list:
    """Pair every deposit leaf of the named caches with its declaration.

    A leaf whose uid the deposit lacks (a hole, a cell outside the deposit's
    b slice) is not a job. A leaf two caches share is one job, under the
    first cache in cache_list.

    Args:
        records (dict): record key -> record (load_records)
        cache_list (list[str]): PAPER_CACHES names to draw from
        source_list (list[str] | None): data sources to keep ('wgn', 'hcp');
            None keeps both

    Returns:
        job_list (list[Job]): in CONFIG grid order
    """
    key_of_uid = {rec['uid']: key for key, rec in records.items()
                  if rec.get('uid')}
    job_list, seen = [], set()
    for name in cache_list:
        b_keep = PAPER_CACHES[name]
        kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc = \
            config.CONFIG[name]
        fnc_name = raw_fnc(fnc).__qualname__
        for kwargs_data in kwargs_data_list:
            if source_list and kwargs_data.get('source') not in source_list:
                continue
            if b_keep is not None and kwargs_data_b(kwargs_data) != b_keep:
                continue
            for kwargs_effect in kwargs_effect_list:
                cell_uid = results.cell_parent_uid(kwargs_data, kwargs_effect)
                if cell_uid not in key_of_uid:
                    continue
                uid_list = results.cell_leaf_uids(
                    kwargs_data, kwargs_effect, kwargs_fnc_list, fnc)
                for kwargs_fnc, uid in zip(kwargs_fnc_list, uid_list):
                    if uid not in key_of_uid or uid in seen:
                        continue
                    seen.add(uid)
                    job_list.append(Job(
                        cache=name, kwargs_data=kwargs_data,
                        kwargs_effect=kwargs_effect, kwargs_fnc=kwargs_fnc,
                        fnc=fnc, key=key_of_uid[uid],
                        cell_key=key_of_uid[cell_uid], cell_uid=cell_uid,
                        shared=_shared_key(fnc_name, kwargs_fnc)))
    return job_list


def draw_order(job_list: list, seed: int) -> list:
    """Shuffle each cache's jobs by seed, then interleave caches in turn.

    Args:
        job_list (list[Job]): index_jobs output
        seed (int): the shuffle's seed

    Returns:
        order (list[Job]): every job once, caches alternating
    """
    rng = random.Random(seed)
    by_cache = collections.defaultdict(list)
    for job in job_list:
        by_cache[job.cache].append(job)
    queue_list = []
    for name in sorted(by_cache):
        queue = by_cache[name]
        rng.shuffle(queue)
        queue_list.append(collections.deque(queue))
    rng.shuffle(queue_list)
    order = []
    while queue_list:
        queue_list = [q for q in queue_list if q]
        for queue in queue_list:
            order.append(queue.popleft())
    return order


class CostModel:
    """Estimate a job's wall clock from the recorded times of this deposit.

    A job pays its own recorded time, plus its cell's build when that cell is
    not the one in memory, plus its shared intermediate when that is not warm
    (priced as the slowest sibling leaf, the one that computed it). The sum is
    scaled by the median observed / estimated ratio, so a slower machine
    learns its own speed after a few jobs.
    """

    def __init__(self, records: dict, job_list: list):
        self.records = records
        self.shared_cost = collections.defaultdict(float)
        for job in job_list:
            if job.shared is not None:
                group = (job.cell_uid, job.shared)
                self.shared_cost[group] = max(self.shared_cost[group],
                                              self.leaf_sec(job))
        self.ratio_list = []

    def leaf_sec(self, job: Job) -> float:
        """Return the leaf's own recorded time_sec."""
        return float(self.records[job.key].get('time_sec') or 0.0)

    def estimate(self, job: Job, warm_cell: str, warm_shared) -> float:
        """Return the job's unscaled cost given what is already in memory.

        Args:
            job (Job): the job to price
            warm_cell (str): uid of the cell currently built, or None
            warm_shared (tuple | None): (cell_uid, shared) now warm

        Returns:
            sec (float): estimated seconds before scaling
        """
        sec = self.leaf_sec(job)
        group = (job.cell_uid, job.shared)
        if job.shared is not None and group != warm_shared:
            sec = max(sec, self.shared_cost[group])
        if job.cell_uid != warm_cell:
            sec += self.cell_sec(job)
        return sec

    def cell_sec(self, job: Job) -> float:
        """Return the recorded time_sec of the job's cell build."""
        return float(self.records[job.cell_key].get('time_sec') or 0.0)

    def rebuild_sec(self, job: Job) -> float:
        """Return what making the job's cell and intermediate warm costs."""
        shared = (self.shared_cost[(job.cell_uid, job.shared)]
                  if job.shared is not None else 0.0)
        return self.cell_sec(job) + shared

    @property
    def speed(self) -> float:
        """Median observed / estimated ratio so far (1 before any job)."""
        return float(np.median(self.ratio_list)) if self.ratio_list else 1.0

    def observe(self, est: float, got: float) -> None:
        """Fold one finished job's ratio in (skipping sub-second ones)."""
        if est >= 1.0:
            self.ratio_list.append(got / est)


def diff(want, got, rtol: float, path: str = '') -> list:
    """List where a replayed snapshot departs from the recorded one.

    Args:
        want: recorded value (_cell form)
        got: replayed value (_cell form)
        rtol (float): relative tolerance a float may move and stay close
        path (str): key path so far

    Returns:
        diff_list (list): [path, want, got, 'close' | 'differs'] per leaf
            value that is not bitwise equal
    """
    if isinstance(want, dict) and isinstance(got, dict):
        out = []
        for k in sorted(set(want) | set(got)):
            if k not in want or k not in got:
                out.append([f'{path}.{k}', want.get(k), got.get(k),
                            'differs'])
            else:
                out += diff(want[k], got[k], rtol, f'{path}.{k}')
        return out
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return [[f'{path}.len', len(want), len(got), 'differs']]
        out = []
        for i, (w, g) in enumerate(zip(want, got)):
            out += diff(w, g, rtol, f'{path}.{i}')
        return out
    if want == got or (isinstance(want, float) and isinstance(got, float)
                       and np.isnan(want) and np.isnan(got)):
        return []
    numeric = (int, float)
    if (isinstance(want, float) or isinstance(got, float)) and \
            isinstance(want, numeric) and isinstance(got, numeric) and \
            not isinstance(want, bool) and not isinstance(got, bool) and \
            np.isclose(want, got, rtol=rtol, atol=0):
        return [[path, want, got, 'close']]
    return [[path, want, got, 'differs']]


@contextlib.contextmanager
def sealed_store():
    """Raise on any read or write of the joblib cache or the records.

    The glow_fit_for_prune it seals is swapped for an in-memory one-entry
    memo over the raw function, so a cell's prune rules share one fit.
    """
    def refuse(*args, **kwargs):
        raise RuntimeError('validate_records reached the benchmark store; '
                           'a replay must run every step unwrapped')

    fit_raw = raw_fnc(run.glow_fit_for_prune)
    memo = {}

    def fit_memo(exp, **kwargs):
        key = repr(sorted((k, repr(v)) for k, v in kwargs.items()
                          if k != 'fit_params'))
        if key not in memo:
            memo.clear()
            memo[key] = fit_raw(exp, **kwargs)
        return memo[key]

    with mock.patch.object(joblib.memory.MemorizedFunc, '__call__', refuse), \
            mock.patch.object(RECORDER, '_store', refuse), \
            mock.patch.object(run, 'glow_fit_for_prune', fit_memo):
        yield


class Replayer:
    """Recompute jobs, holding the last realized cell in memory.

    Attributes:
        rtol (float): float tolerance for a close verdict
        warm_cell (str | None): uid of the cell in memory
        warm_shared (tuple | None): (cell_uid, shared) last computed
    """

    def __init__(self, records: dict, rtol: float):
        self.records = records
        self.rtol = rtol
        self.warm_cell = None
        self.warm_shared = None
        self._cell = None
        self._get_exp_effect = raw_fnc(get_exp_effect)

    def replay(self, job: Job) -> dict:
        """Recompute one job and diff it against its records.

        Args:
            job (Job): the leaf to recompute

        Returns:
            row (dict): {cache, function, key, cell_key, status, diff,
                time_rec, time_now, cell_time_now?, time_ratio?, fit_params?,
                error?}
        """
        rec = self.records[job.key]
        row = dict(cache=job.cache, function=rec['function'], key=job.key,
                   cell_key=job.cell_key)
        if 'fit_params' in rec.get('inputs', {}):
            row['fit_params'] = rec['inputs']['fit_params']
        diff_list = []
        try:
            if job.cell_uid != self.warm_cell:
                self.warm_cell, self._cell = None, None
                t0 = time.perf_counter()
                cell = self._get_exp_effect(job.kwargs_data,
                                            job.kwargs_effect)
                row['cell_time_now'] = time.perf_counter() - t0
                cell_want = self.records[job.cell_key]['outputs']['cell']
                diff_list += diff(cell_want, repr(cell), self.rtol, 'cell')
                self.warm_cell, self._cell = job.cell_uid, cell
            leaf = raw_fnc(job.fnc)
            t0 = time.perf_counter()
            out = leaf(self._cell, parent_uid=job.cell_uid, **job.kwargs_fnc)
            row['time_now'] = time.perf_counter() - t0
            if job.shared is not None:
                self.warm_shared = (job.cell_uid, job.shared)
            (name, want), = rec['outputs'].items()
            diff_list += diff(want, _cell(out), self.rtol, name)
        except Exception as exc:
            row['status'] = 'error'
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['traceback'] = traceback.format_exc()
            return row
        row['time_rec'] = rec.get('time_sec')
        if rec['function'] in TIME_FNC and rec.get('time_sec'):
            row['time_ratio'] = row['time_now'] / rec['time_sec']
        row['diff'] = diff_list
        kinds = {d[3] for d in diff_list}
        row['status'] = ('differs' if 'differs' in kinds
                         else 'close' if kinds else 'identical')
        return row


def validate(order: list, records: dict, budget_sec: float, rtol: float,
             log_path: pathlib.Path) -> list:
    """Replay jobs in order until the budget is spent, logging each.

    A job whose estimate overruns the remaining budget is passed over, so
    cheaper ones later in the order still run. After each job, unrun jobs on
    the same cell whose marginal cost is below what rebuilding that cell (and
    its shared intermediate) would cost run next. A running job is never cut
    short, so the last one can overrun by its own length.

    Args:
        order (list[Job]): draw_order output
        records (dict): record key -> record
        budget_sec (float): wall-clock budget
        rtol (float): float tolerance for close
        log_path (pathlib.Path): JSONL log, one row per job (Replayer.replay)

    Returns:
        row_list (list[dict]): the rows logged
    """
    cost = CostModel(records, order)
    replayer = Replayer(records, rtol)
    by_cell = collections.defaultdict(list)
    for job in order:
        by_cell[job.cell_uid].append(job)
    done = set()
    row_list = []
    t_start = time.perf_counter()

    def run_one(job, est):
        est_scaled = est * cost.speed
        row = replayer.replay(job)
        done.add(job.key)
        spent = row.get('time_now', 0.0) + row.get('cell_time_now', 0.0)
        cost.observe(est, spent)
        row['elapsed'] = time.perf_counter() - t_start
        row_list.append(row)
        with open(log_path, 'a') as handle:
            handle.write(json.dumps(row) + '\n')
        n_diff = len(row.get('diff', ()))
        print(f'[{row["elapsed"]:8.0f}s] {row["status"]:9s} '
              f'{job.cache:28s} {row["function"]:18s} {job.key[:12]} '
              f'est {est_scaled:7.1f}s got {spent:7.1f}s'
              + (f' ({n_diff} diff)' if n_diff else '')
              + (f' {row["error"]}' if 'error' in row else ''), flush=True)

    for job in order:
        if job.key in done:
            continue
        est = cost.estimate(job, replayer.warm_cell, replayer.warm_shared)
        left = budget_sec - (time.perf_counter() - t_start)
        if left <= 0:
            break
        if est * cost.speed > left:
            continue
        run_one(job, est)
        for sib in by_cell[job.cell_uid]:
            if sib.key in done:
                continue
            marginal = cost.estimate(sib, replayer.warm_cell,
                                     replayer.warm_shared)
            left = budget_sec - (time.perf_counter() - t_start)
            if (marginal < cost.rebuild_sec(sib)
                    and marginal * cost.speed <= left):
                run_one(sib, marginal)
    return row_list


def summarize(row_list: list, job_list: list) -> str:
    """Return a per-cache table of verdicts against the leaves available."""
    n_avail = collections.Counter(job.cache for job in job_list)
    count = collections.defaultdict(collections.Counter)
    for row in row_list:
        count[row['cache']][row['status']] += 1
    status_list = ('identical', 'close', 'differs', 'error')
    lines = [f'{"cache":28s} {"avail":>6s} {"run":>5s} '
             + ' '.join(f'{s:>9s}' for s in status_list)]
    for name in sorted(n_avail):
        c = count[name]
        lines.append(f'{name:28s} {n_avail[name]:6d} {sum(c.values()):5d} '
                     + ' '.join(f'{c[s]:9d}' for s in status_list))
    return '\n'.join(lines)


def main(argv=None) -> None:
    """Index the deposit, replay a seeded sample within budget, report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seed', type=int, required=True,
                        help='seed of the random draw order')
    parser.add_argument('--runtime', type=parse_runtime, required=True,
                        help="wall-clock budget: seconds, or '30m', '2h', "
                             "'1h30m'")
    parser.add_argument('--records', type=pathlib.Path,
                        default=get_path_records(),
                        help='deposit zip, or a records dir (default: the '
                             'local store)')
    parser.add_argument('--cache', action='append', choices=PAPER_CACHES,
                        help='draw only from this cache (repeatable)')
    parser.add_argument('--source', action='append', choices=['wgn', 'hcp'],
                        help='draw only cells of this source (repeatable); '
                             'wgn needs no imaging data')
    parser.add_argument('--rtol', type=float, default=1e-6,
                        help='relative tolerance for a close float')
    parser.add_argument('--out', type=pathlib.Path, default=None,
                        help='JSONL log (default: results/validate_records/)')
    parser.add_argument('--dry-run', action='store_true',
                        help='index and price the deposit, run nothing')
    args = parser.parse_args(argv)

    if config._N_SEED_CAP:
        print(f'warning: GLOW_BENCH_N_SEED={config._N_SEED_CAP} caps the '
              f'CONFIG seed grids, so leaves past it cannot be drawn')

    records = load_records(args.records)
    cache_list = args.cache or list(PAPER_CACHES)
    job_list = index_jobs(records, cache_list, args.source)
    n_leaf = collections.Counter(job.cache for job in job_list)
    print(f'{len(records)} records from {args.records}; {len(job_list)} '
          f'replayable leaves')

    if args.dry_run:
        cost = CostModel(records, job_list)
        for name in sorted(n_leaf):
            jobs = [j for j in job_list if j.cache == name]
            cold = [cost.estimate(j, None, None) for j in jobs]
            print(f'  {name:28s} {n_leaf[name]:6d} leaves, cold replay '
                  f'median {np.median(cold):7.1f}s max {max(cold):7.1f}s')
        return

    if args.out is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = get_path_result() / 'validate_records'
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = out_dir / f'seed{args.seed}_{stamp}.jsonl'
    print(f'log: {args.out}')

    order = draw_order(job_list, args.seed)
    with sealed_store():
        row_list = validate(order, records, args.runtime, args.rtol, args.out)
    print(summarize(row_list, job_list))
    bad = [r for r in row_list if r['status'] in ('differs', 'error')]
    if bad:
        print(f'{len(bad)} leaf/leaves differ or failed; see {args.out}')
        sys.exit(1)


if __name__ == '__main__':
    main()
