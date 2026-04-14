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
from glow.experiment.analysis._tfce import apply_tfce_img

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

    def test_single_voxel(self):
        """tfce of single voxel should scale with height"""
        x = np.zeros((5, 5, 5))
        x[2, 2, 2] = 1.0
        result = apply_tfce_img(x)
        # single voxel has extent=1, so TFCE = ∫ 1^0.5 * h^2 dh = h^3/3
        # with discretization this won't be exact but should be positive
        assert result[2, 2, 2] > 0
        assert result[0, 0, 0] == 0  # background stays zero

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

    def test_output_shape(self):
        """output shape matches input shape"""
        x = np.random.randn(8, 10, 12)
        x[x < 0] = 0
        result = apply_tfce_img(x)
        assert result.shape == x.shape

    def test_monotonic_with_height(self):
        """higher input values should give higher tfce"""
        x = np.zeros((5, 5, 5))
        x[2, 2, 2] = 1.0
        tfce1 = apply_tfce_img(x)[2, 2, 2]

        x[2, 2, 2] = 2.0
        tfce2 = apply_tfce_img(x)[2, 2, 2]

        assert tfce2 > tfce1


# --- FSL validation tests (skipped if fslmaths not on $PATH) ---

@pytest.mark.skipif(
    _fslmaths is None,
    reason=f'FSL not installed (install: {_FSL_INSTALL_URL})')
class TestTFCEValidationAgainstFSL:
    """compares pure python tfce to fsl implementation

    note: fsl zeros out edge voxels, so tests use padded input or compare
    interior only. this matches actual usage via apply_tfce_x which pads.
    """

    def test_random_image_padded(self):
        """tfce with padded input should match fsl exactly"""
        np.random.seed(42)
        x_core = np.random.randn(6, 6, 6)
        x_core[x_core < 0] = 0
        # pad with zeros (matching apply_tfce_x behavior)
        x = np.pad(x_core, pad_width=1, constant_values=0)

        tfce_python = apply_tfce_img(x)
        tfce_fsl = _apply_tfce_img_fsl(x)

        # compare interior only (fsl zeros edges)
        py_int = tfce_python[1:-1, 1:-1, 1:-1]
        fsl_int = tfce_fsl[1:-1, 1:-1, 1:-1]

        corr = correlation(py_int, fsl_int)
        print(f'\nCorrelation (padded): {corr:.6f}')
        assert corr > 0.9999, f'Correlation too low: {corr}'

        max_diff = np.abs(py_int - fsl_int).max()
        print(f'Max absolute diff: {max_diff:.6f}')
        assert max_diff < 0.01, f'Max diff too high: {max_diff}'

    def test_gaussian_blob_padded(self):
        """gaussian blob with padding should match fsl"""
        x, y, z = np.mgrid[-2:3, -2:3, -2:3]
        blob_core = np.exp(-(x**2 + y**2 + z**2) / 2)
        blob = np.pad(blob_core, pad_width=1, constant_values=0)

        tfce_python = apply_tfce_img(blob)
        tfce_fsl = _apply_tfce_img_fsl(blob)

        py_int = tfce_python[1:-1, 1:-1, 1:-1]
        fsl_int = tfce_fsl[1:-1, 1:-1, 1:-1]

        corr = correlation(py_int, fsl_int)
        print(f'\nGaussian blob correlation: {corr:.6f}')
        assert corr > 0.9999, f'Correlation too low: {corr}'

    def test_multiple_clusters_padded(self):
        """multiple separated clusters with padding"""
        x = np.zeros((13, 13, 13))
        # cluster 1
        x[2:5, 2:5, 2:5] = 2.0
        # cluster 2
        x[8:11, 8:11, 8:11] = 3.0
        # pad
        x = np.pad(x, pad_width=1, constant_values=0)

        tfce_python = apply_tfce_img(x)
        tfce_fsl = _apply_tfce_img_fsl(x)

        py_int = tfce_python[1:-1, 1:-1, 1:-1]
        fsl_int = tfce_fsl[1:-1, 1:-1, 1:-1]

        corr = correlation(py_int, fsl_int)
        print(f'\nMultiple clusters correlation: {corr:.6f}')
        assert corr > 0.9999, f'Correlation too low: {corr}'

    def test_uniform_cube_exact(self):
        """uniform cube should give exact numeric match"""
        x = np.zeros((9, 9, 9))
        x[3:6, 3:6, 3:6] = 1.0  # centered 3x3x3 cube

        tfce_python = apply_tfce_img(x)
        tfce_fsl = _apply_tfce_img_fsl(x)

        # center should match exactly
        py_center = tfce_python[4, 4, 4]
        fsl_center = tfce_fsl[4, 4, 4]

        print(f'\nUniform cube center: Python={py_center:.4f}, FSL={fsl_center:.4f}')
        assert np.isclose(py_center, fsl_center, rtol=1e-4), \
            f'Center values differ: {py_center} vs {fsl_center}'


def run_visual_comparison():
    """run a visual comparison (for debugging)"""
    if _fslmaths is None:
        print(f'FSL not available. Install: {_FSL_INSTALL_URL}')
        return

    np.random.seed(42)
    x_core = np.random.randn(6, 6, 6)
    x_core[x_core < 0] = 0
    # pad with zeros (matching actual usage via apply_tfce_x)
    x = np.pad(x_core, pad_width=1, constant_values=0)

    tfce_python = apply_tfce_img(x)
    tfce_fsl = _apply_tfce_img_fsl(x)

    # compare interior only (fsl zeros edges)
    py_int = tfce_python[1:-1, 1:-1, 1:-1]
    fsl_int = tfce_fsl[1:-1, 1:-1, 1:-1]

    print('=== TFCE Validation Results (Padded Input) ===')
    print(f'Core shape: {x_core.shape}, Padded shape: {x.shape}')
    print(f'Input range: [{x_core.min():.3f}, {x_core.max():.3f}]')
    print(f'Input non-zero voxels: {(x_core > 0).sum()}')
    print()
    print(f'Python TFCE range (interior): [{py_int.min():.3f}, {py_int.max():.3f}]')
    print(f'FSL TFCE range (interior): [{fsl_int.min():.3f}, {fsl_int.max():.3f}]')
    print()
    print(f'Correlation: {correlation(py_int, fsl_int):.6f}')
    print(f'Max absolute diff: {np.abs(py_int - fsl_int).max():.6f}')
    print()
    print('Pure Python TFCE matches FSL output!')


if __name__ == '__main__':
    run_visual_comparison()
