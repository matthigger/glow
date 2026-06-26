"""Fit one Analysis on one Experiment and read out its common outputs.

The low-level benchmark primitive: ``run_ana`` takes an already-built
Experiment and an unfitted Analysis recipe, runs ``ana.fit(exp)``, and
returns the subset of fitted attributes that *every* Analysis exposes --
no data building, no effect planting (the paper layer wraps this with
those). It is memoised + recorded exactly like the data.py builders: it
shares their MEMORY / RECORDER, so a fit is cached on disk and joins the
same provenance DAG -- a run_ana record's ``exp`` input links to the
build (data_factory, or effect_factory's plant) that produced it, so
``RECORDER.flatten_to_df`` chains data -> (plant ->) analysis into one
row.

What "every Analysis exposes" means is fixed by the Analysis ABC, which
declares exactly two fitted outputs (see glow.analysis._base.Analysis):

    effect_list  list[Effect]   discovered regions; each carries a .mask
                                (GLOW's EffectEstimate also .reg_idx /
                                .pval_fwer, but .mask is the only field
                                common to all)
    pval         np.array       FWER-controlled p-values, 1-D -- length
                                num_reg for GLOW, num_vox for VBA / CET,
                                but the same role and dtype either way

These are the "outputs" only in the read-them-off-the-object sense: an
Analysis is fit-in-place (it returns self, sklearn-style), so the result
lives in its attributes. The richer per-method attributes (GLOW's llr /
z / children / max_z_null, the voxel methods' stat, CET's cft) are
deliberately *not* returned -- they are not part of the cross-method
contract, so a benchmark comparing heterogeneous methods cannot assume
them.
"""

import copy

from glow.analysis import Analysis
from glow.experiment.exper import Experiment

# share the data.py builders' disk cache + recorder, so a fit is memoised
# beside the builds and run_ana joins their provenance DAG (see module docs).
from .data import MEMORY, RECORDER


@MEMORY.cache
@RECORDER(output_name_list=['effect_list', 'pval'])
def run_ana(exp: Experiment, ana: Analysis):
    """Fit ``ana`` on ``exp`` and return its common (cross-method) outputs.

    Calls ``ana.fit(exp)`` (every Analysis scales exp on the way in and
    returns self), then returns the two attributes the Analysis ABC
    guarantees on any fitted analysis -- the uniform surface a benchmark
    can compare across GLOW, VBA, CET, etc. without knowing the concrete
    type.

    Memoised on disk (MEMORY) with the recorder nested inside the cache,
    keyed by joblib's hash of (exp, ana): a repeated (exp, ana) pair is
    served from the cache and only a real (cache-miss) fit is recorded.
    Two distinct recipes hash distinctly, so each variant caches and
    records on its own.

    ``ana`` is never mutated -- fit runs on a private copy, leaving the
    caller's recipe (and so its hash) untouched. That is what makes the
    cache key stable when the same recipe object is reused, and the
    result deterministic in (exp, ana). The fitted copy is discarded, so
    a method's extra attributes (GLOW's llr / z, the voxel methods'
    stat, ...) are not recoverable here; call ``ana.fit(exp)`` directly
    (uncached) for those.

    Args:
        exp (Experiment): the experiment to analyze (raw or already
            scaled; fit idempotently scales it).
        ana (Analysis): an unfitted analysis recipe (its __init__ config
            knobs only -- the experiment is not stored on it).

    Returns:
        effect_list (list): the discovered Effect objects.
        pval (np.array): FWER-controlled p-values (num_reg for GLOW,
            num_vox for the voxel methods).
    """
    ana = copy.deepcopy(ana)
    ana.fit(exp)
    return ana.effect_list, ana.pval
