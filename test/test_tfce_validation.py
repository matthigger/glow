"""validates pure python tfce against fsl implementation

FSL is only needed for the validation tests (TestTFCEValidationAgainstFSL).
Install FSL: https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html
"""

import os
import subprocess
import tempfile
from shutil import which

import nibabel as nib
import numpy as np
import pytest
from glow.analysis.vba._tfce import apply_tfce_img

# --- FSL detection (finds fslmaths on $PATH) ---

_FSL_INSTALL_URL = 'https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html'

_fslmaths = which('fslmaths')


def _apply_tfce_img_fsl(x):
    """applies tfce using fsl's fslmaths (for validation only)

    Raises:
        RuntimeError: if fslmaths is not on $PATH
    """
    if _fslmaths is None:
        raise RuntimeError(
            f'fslmaths not found on $PATH. '
            f'Install FSL: {_FSL_INSTALL_URL}'
        )

    # get input / output files
    f = tempfile.NamedTemporaryFile(suffix='.nii.gz').name
    f_x_in = f.replace('.nii', '_x.nii')
    f_out = f.replace('.nii', '_x_tfce.nii')

    # write input to disk
    img = nib.Nifti1Image(x, affine=np.eye(4))
    img.to_filename(f_x_in)

    cmd = f'{_fslmaths} {f_x_in} -tfce 2 .5 6 {f_out}'
    proc = subprocess.run(cmd, shell=True, capture_output=True,
                          executable='/bin/bash')
    assert not proc.returncode, proc.stderr

    x_tfce = nib.load(f_out).get_fdata()

    # cleanup
    os.remove(f_out)
    os.remove(f_x_in)

    return x_tfce


# --- helpers ---

def correlation(a, b):
    """compute pearson correlation between two arrays"""
    a_flat = a.ravel()
    b_flat = b.ravel()
    return np.corrcoef(a_flat, b_flat)[0, 1]


def relative_error(a, b):
    """compute mean relative error"""
    mask = b != 0
    if not mask.any():
        return 0.0
    return np.mean(np.abs(a[mask] - b[mask]) / np.abs(b[mask]))


# --- pure python tests (no FSL needed) ---

class TestTFCEPurePython:
    """tests for pure python tfce implementation"""

    def test_zeros_input(self):
        """tfce of zeros should be zeros"""
        x = np.zeros((5, 5, 5))
        result = apply_tfce_img(x)
        assert np.allclose(result, 0)

    def test_negative_input(self):
        """tfce of all negative values should be zeros"""
        x = -np.ones((5, 5, 5))
        result = apply_tfce_img(x)
        assert np.allclose(result, 0)

    def test_cluster_enhancement(self):
        """larger clusters should have higher tfce than single voxels"""
        # single voxel
        x1 = np.zeros((7, 7, 7))
        x1[3, 3, 3] = 1.0
        tfce1 = apply_tfce_img(x1)

        # 3x3x3 cluster (27 voxels)
        x2 = np.zeros((7, 7, 7))
        x2[2:5, 2:5, 2:5] = 1.0
        tfce2 = apply_tfce_img(x2)

        # cluster center should have higher tfce than single voxel
        assert tfce2[3, 3, 3] > tfce1[3, 3, 3]

    def test_monotonic_with_height(self):
        """higher input values should give higher (and positive) tfce"""
        x = np.zeros((5, 5, 5))
        x[2, 2, 2] = 1.0
        result1 = apply_tfce_img(x)
        tfce1 = result1[2, 2, 2]

        x[2, 2, 2] = 2.0
        tfce2 = apply_tfce_img(x)[2, 2, 2]

        # monotone in height -- this also subsumes positivity (0 < tfce1 < tfce2)
        assert 0 < tfce1 < tfce2
        # background voxels stay zero
        assert result1[0, 0, 0] == 0


# --- FSL validation tests (skipped if fslmaths not on $PATH) ---

@pytest.mark.skipif(
    _fslmaths is None,
    reason=f'FSL not installed (install: {_FSL_INSTALL_URL})')
class TestTFCEValidationAgainstFSL:
    """compares pure python tfce to fsl implementation

    note: fsl zeros out edge voxels, so tests use padded input or compare
    interior only. this matches actual usage via apply_tfce_x which pads.
    """

    @staticmethod
    def _padded_input(name):
        """build a zero-padded test image (fsl zeros edges, so we pad)."""
        if name == 'random':
            np.random.seed(42)
            x_core = np.random.randn(6, 6, 6)
            x_core[x_core < 0] = 0
        elif name == 'gaussian_blob':
            x, y, z = np.mgrid[-2:3, -2:3, -2:3]
            x_core = np.exp(-(x**2 + y**2 + z**2) / 2)
        elif name == 'multiple_clusters':
            x_core = np.zeros((13, 13, 13))
            x_core[2:5, 2:5, 2:5] = 2.0   # cluster 1
            x_core[8:11, 8:11, 8:11] = 3.0  # cluster 2
        else:
            raise ValueError(name)
        return np.pad(x_core, pad_width=1, constant_values=0)

    @pytest.mark.parametrize('input', ['random', 'gaussian_blob',
                                       'multiple_clusters'])
    def test_matches_fsl(self, input):
        """padded python tfce should correlate ~exactly with fsl"""
        x = self._padded_input(input)

        tfce_python = apply_tfce_img(x)
        tfce_fsl = _apply_tfce_img_fsl(x)

        # compare interior only (fsl zeros edges)
        py_int = tfce_python[1:-1, 1:-1, 1:-1]
        fsl_int = tfce_fsl[1:-1, 1:-1, 1:-1]

        corr = correlation(py_int, fsl_int)
        assert corr > 0.9999, f'[{input}] correlation too low: {corr:.6f}'

        max_diff = np.abs(py_int - fsl_int).max()
        assert max_diff < 0.01, f'[{input}] max abs diff too high: {max_diff:.6f}'

    def test_uniform_cube_exact(self):
        """uniform cube should give exact numeric match"""
        x = np.zeros((9, 9, 9))
        x[3:6, 3:6, 3:6] = 1.0  # centered 3x3x3 cube

        tfce_python = apply_tfce_img(x)
        tfce_fsl = _apply_tfce_img_fsl(x)

        # center should match exactly
        py_center = tfce_python[4, 4, 4]
        fsl_center = tfce_fsl[4, 4, 4]

        assert np.isclose(py_center, fsl_center, rtol=1e-4), \
            f'center values differ: Python={py_center:.4f} vs FSL={fsl_center:.4f}'
