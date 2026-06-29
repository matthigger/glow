"""Fit one Analysis on one Experiment and score the result.

The low-level benchmark primitive: run_ana takes an already-built
Experiment, an unfitted Analysis recipe, and the planted target(s), runs
ana.fit(exp), and returns the detection score of the discovered effects
against the target -- no data building, no effect planting (the config
layer wraps this with those). It is memoised + recorded exactly like the
data.py builders: it shares their MEMORY / RECORDER, so a run is cached on
disk and joins the same provenance DAG -- a run_ana record's exp input
links to the build (data_factory, or effect_factory's plant) that produced
it, so RECORDER.flatten_to_df chains data -> (plant ->) score into one row.

Scoring is inlined rather than a separate recorded step on purpose. The
fitted Analysis is the heavy object (per-region llr / z / children /
max_z_null arrays, ~tens of MB at scale); used as an in-memory local and
discarded, only the small score dict reaches the cache and the records.
Keeping it a downstream node would force either a bespoke link-typed
result carrier or recording that whole object -- and the planted target is
already baked into exp, so folding the score in adds no redundant refits.
The one method-uniform surface every Analysis exposes (effect_list + pval)
is read inside score_effects; the score dict it returns is what every
method is compared on (see .score).

score_effects records only the four confusion counts (+ per-region
geometry, min_pval); Dice / sensitivity / PPV / specificity derive
downstream, so a later metric change re-derives from the records without
re-fitting -- the bulk of the re-score flexibility a separate step would
have bought, at none of the linking cost.
"""

import copy

from glow.analysis import Analysis
from glow.experiment.exper import Experiment

# share the data.py builders' disk cache + recorder, so a fit is memoised
# beside the builds and run_ana joins their provenance DAG (see module docs).
from .data import MEMORY, RECORDER
from .score import score_effects


@MEMORY.cache(ignore=['label'])
@RECORDER(output_name='score')
def run_ana(exp: Experiment, ana: Analysis, mask_target_list, label=None):
    """Fit ana on exp and score it against the planted target(s).

    Calls ana.fit(exp) (every Analysis scales exp on the way in and returns
    self), then scores the discovered effects against the planted supports
    with score_effects -- the uniform detection score a benchmark compares
    across GLOW, VBA, CET, etc. without knowing the concrete type.

    Memoised on disk (MEMORY) with the recorder nested inside the cache,
    keyed by joblib's hash of (exp, ana, mask_target_list): a repeat is
    served from the cache and only a real (cache-miss) run is recorded.
    Two distinct recipes hash distinctly, so each variant caches and
    records on its own. mask_target_list is a deterministic function of exp
    (the effect was planted into it), so it adds no independent cache key
    axis -- it is there because score_effects needs the realized supports,
    which exp does not itself carry.

    ana is never mutated -- fit runs on a private copy, leaving the caller's
    recipe (and so its hash) untouched. That is what makes the cache key
    stable when the same recipe object is reused, and the result
    deterministic in (exp, ana, mask_target_list). The fitted copy (and its
    heavy per-method arrays) is discarded; only the score dict is returned,
    cached, and recorded.

    label (the config layer's ana_kwargs_dict method name) is unused by the
    computation; it is recorded beside the score (the in.label column) so a
    method is named in the output, and @MEMORY.cache(ignore=['label'])
    drops it from the cache key so renaming a method does not invalidate its
    cache.

    Args:
        exp (Experiment): the experiment to analyze (raw or already
            scaled; fit idempotently scales it).
        ana (Analysis): an unfitted analysis recipe (its __init__ config
            knobs only -- the experiment is not stored on it).
        mask_target_list (list): the planted effect supports, one (X, Y, Z)
            bool mask per EffectSynthetic (effect_factory's mask output);
            empty for the null / FWER-calibration path.
        label (str): the method label recorded beside the score; unused by
            the computation and excluded from the cache key.

    Returns:
        score (dict): the detection score (see .score.score_effects):
            global num_vox / min_pval / n_pred, a per-region pred list, and
            the target (+ per-effect target0..N) confusion blocks.
    """
    ana = copy.deepcopy(ana)
    ana.fit(exp)
    return score_effects(ana, mask_target_list, mask_active=exp.mask_idx > -1)
