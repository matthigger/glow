# Memory Estimation for AWS Batch Workers

With the streaming synthesis pipeline, both permutation and synthesis
workers have the same memory footprint.  Memory scales with experiment
dimensions, not the number of permutations.

## Per-worker memory model

Each worker holds the experiment data tensor `(b, num_img, num_vox)`,
a permuted copy, and working memory for clustering/stats (~1x
overhead):

    est_mb = 3 × b × num_img × num_vox × 8 / 1024²

where `num_img` is the number of subjects (scans) in the analysis and
`num_vox` is the number of voxels in the brain mask.  This is
implemented in `AWSBatchRunner.estimate_memory_mb`.

## Maximum subjects per tier

Assuming `b=1` and a whole-brain mask volume of **1450 cm³**
(1,450,000 mm³), so `num_vox = 1,450,000 / resolution³`:

| Resolution | Voxels      | 2 GB  | 4 GB  | 8 GB  | 16 GB  |
|------------|-------------|-------|-------|-------|--------|
| 2.00 mm    | 181,000     | 482   | 964   | 1,928 | 3,856  |
| 1.50 mm    | 430,000     | 203   | 406   | 813   | 1,627  |
| 1.25 mm    | 742,000     | 117   | 235   | 470   | 941    |
| 1.00 mm    | 1,450,000   | 60    | 120   | 241   | 482    |
| 0.80 mm    | 2,830,000   | 30    | 61    | 123   | 246    |

Each cell is the maximum `num_img` (subjects) that fits in that tier
at that resolution.  For values `b>1`, just divide the table entries above by `b`.  For example, at a 2 mm resolution with 2 GB of memory we can analyze 482 / 2 = 241 subjects when `b=2`.

## OOM tier escalation

Jobs start at the **2 GB** default (1 vCPU : 2 GiB on AWS compute
instances).  OOM retries escalate through **4 GB → 8 GB → 16 GB**.
If a job exceeds the 16 GB ceiling, a `MemoryError` is raised and the
run is stopped.
