"""Strict FWER calibration and exchangeability under the null.

Gated by --runslow: the whole file. A calibration test is a claim about a
rejection RATE, and a rate estimated from a few dozen null trials cannot
separate correct control from a doubling of it -- at K=40 with a budget of
0.20 (what this suite used to run in every pytest invocation) the power
against 2x inflation is about 1.5%. Rather than keep a fast test that
certifies nothing, the trial count is raised until the claim is real and
the cost is paid only on demand.

Sizing, for nominal alpha = 0.05 and K = 1000 (exact binomial, one-sided):

    P(rate > 0.068 | true = 0.05)  <  0.01   false alarms
    P(rate > 0.068 | true = 0.075) ~= 0.78   catches 1.5x inflation
    P(rate > 0.068 | true = 0.10)  ~= 0.9998 catches 2x inflation

Every arm the paper reports is covered, since FWER control is a property
of the arm and not of the package: GLOW, VBA, VBA+z, VBA-TFCE and CET.

Run:
    ~/venv_glow/bin/pytest test/test_fwer_calibration.py --runslow -v
"""

import numpy as np
import pytest
from joblib import Parallel, delayed
from scipy import stats as sp_stats

from glow.analysis import AnalysisCET, AnalysisGLOW, AnalysisVBA
from glow.analysis.mancova import get_wilks
from glow.experiment import Experiment


pytestmark = pytest.mark.slow

K = 1_000
N_PERM = 49
ALPHA = 0.05

# exact one-sided 99% binomial upper bound at K=1000, alpha=0.05 is 0.067;
# 0.068 clears it, so a correctly-calibrated arm fails below 1% of the time
MAX_RATE = 0.068

# the exchangeability check ranks one observed draw among N_PERM permutations,
# so it needs fewer trials than a rate estimate to have power
K_RANK = 500


def _null_exp(seed: int):
    """Build one pure-noise (H0) experiment."""
    return Experiment.from_gauss(a=2, b=1, shape=(5, 5), num_img=30,
                                 seed=seed)


def _rejects(make_ana, seed: int) -> bool:
    """Whether one H0 fit declares any effect significant."""
    return len(make_ana().fit(_null_exp(seed), n_jobs=1).effect_list) > 0


def _null_rejection_rate(make_ana, k: int = K) -> float:
    """Share of k null experiments in which H0 is rejected.

    Parallel over seeds with each fit held to one core: the seeds are
    independent and every fit is small, so this is the level worth
    parallelising (an inner n_jobs would nest joblib pools).

    Args:
        make_ana (Callable): zero-arg factory returning a fresh recipe
        k (int): number of null trials

    Returns:
        float: rejection rate in [0, 1]
    """
    hits = Parallel(n_jobs=-1)(
        delayed(_rejects)(make_ana, seed) for seed in range(k))
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# FWER calibration, every reported arm
# ---------------------------------------------------------------------------

ARMS = {
    'GLOW': lambda: AnalysisGLOW(n_perm_fwer=N_PERM, alpha_fwer=ALPHA,
                                 min_vox=1),
    'VBA': lambda: AnalysisVBA(n_perm_fwer=N_PERM, alpha_fwer=ALPHA),
    'VBA+z': lambda: AnalysisVBA(n_perm_fwer=N_PERM, alpha_fwer=ALPHA,
                                 z_flag=True),
    'VBA-TFCE': lambda: AnalysisVBA(n_perm_fwer=N_PERM, alpha_fwer=ALPHA,
                                    tfce_flag=True),
    'CET': lambda: AnalysisCET(n_perm_fwer=N_PERM, alpha_fwer=ALPHA),
}


@pytest.mark.parametrize('label', list(ARMS))
def test_fwer_controlled_under_null(label):
    """Under H0 the rejection rate stays within binomial noise of alpha.

    VBA+z carries extra history: when z_score_stat took mu/sigma from the
    null rows only, the observed row was standardized by parameters it did
    not contribute to, which broke exchangeability at finite B and inflated
    rejection to ~11% at B=49 against a nominal 5% (Phipson & Smyth 2010;
    Winkler et al. 2014). That regression would fail this budget outright.
    """
    rate = _null_rejection_rate(ARMS[label])
    assert rate <= MAX_RATE, (
        f'{label} FWER inflated: {rate:.3f} ({int(rate * K)}/{K} rejected) '
        f'exceeds the {MAX_RATE} budget for a nominal {ALPHA}')


# ---------------------------------------------------------------------------
# Permutation exchangeability
# ---------------------------------------------------------------------------

def _observed_rank(seed: int) -> int:
    """Rank of the observed max-stat among all N_PERM + 1 draws.

    Under H0 the observed draw is exchangeable with the permuted ones, so
    this rank is uniform on 1..N_PERM+1.

    Args:
        seed (int): null-experiment seed

    Returns:
        int: rank in 1..N_PERM+1, 1 when the observed draw is the largest
    """
    exp = _null_exp(seed)
    ana = AnalysisVBA(n_perm_fwer=N_PERM, get_stat=get_wilks)
    stat = np.full((N_PERM + 1, exp.y.shape[2]), np.nan)
    for k in range(N_PERM + 1):
        stat[k, :] = ana.get_stat_perm(exp.permute(k) if k else exp)
    max_stat = np.nanmax(stat, axis=1)
    return int(np.sum(max_stat >= max_stat[0]))


def test_observed_rank_uniform():
    """The observed max-stat rank is uniform over the permutation draws.

    Chi-square against the DISCRETE uniform on 1..N_PERM+1, which is the
    rank's actual null. The KS test this replaces mapped the rank to
    (r - 0.5)/(N_PERM + 1) and compared it to a CONTINUOUS uniform, which
    misreads the statistic in the anti-conservative direction: between
    atoms the empirical CDF is flat while the reference CDF keeps rising,
    so even a perfectly balanced sample scores D = 1/(2 (N_PERM + 1)).
    That floor is independent of K while the critical value falls as
    1/sqrt(K), so the over-rejection worsens as trials are added --
    measured false-positive rate at a nominal 0.01 was 0.013 at K=50 and
    0.024 at K=500.

    Power was the bigger problem, and it is why K rose to 500: against an
    alternative that hands the observed draw the maximum 5% of the time,
    the old K=50 KS test rejected 3% of the time, this one 80%.
    """
    ranks = Parallel(n_jobs=-1)(
        delayed(_observed_rank)(seed) for seed in range(K_RANK))

    counts = np.bincount(np.asarray(ranks), minlength=N_PERM + 2)[1:]
    assert counts.sum() == K_RANK, 'a rank fell outside 1..N_PERM+1'

    # K_RANK / (N_PERM + 1) = 10 expected per cell, comfortably above the
    # count of 5 that chi-square's asymptotic null needs
    _chi2, p = sp_stats.chisquare(counts)
    assert p > 0.001, (
        f'observed max-stat rank is not uniform (chi-square p={p:.4g}), '
        f'suggesting broken exchangeability')
