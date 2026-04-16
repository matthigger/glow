#!/usr/bin/env python3
"""Test script for worker memory profiling

Simulates worker execution and triggers memory profiling to test the output.
"""

import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import glow
from glow.benchmark.config import Config
from glow.benchmark.runner import RunAna
from glow.aws.worker import print_memory_profile, get_memory_profile


def bench_memory_profile():
    """Test memory profiling with a real config"""
    print("=" * 60)
    print("Testing Worker Memory Profiling")
    print("=" * 60)
    
    # Create a simple WGN config (fast to set up)
    ana_kwargs_dict = {
        'GLOW': (glow.analysis.AnalysisGLOW, {
            'n_perm_fwer': 50,
            'min_size': 1,
            'alpha_fwer': 0.05,
            'n_jobs_perm': 1
        })
    }
    
    config = Config(
        label='test_memory_profile',
        source='wgn',
        runner=RunAna(ana_kwargs_dict),
        wgn_shape=(5, 5, 5),  # 125 voxels
        wgn_a=2,
        wgn_b=2,
        wgn_num_img=10,
        exp_seed=0,
        n_seed=1,
        effect_llr_all=np.array([0.05]),
        effect_perc=0.2,
        n_jobs=1,
        detail_save=False,
        error_save=False
    )
    
    # Set up temp folder
    temp_folder = Path(tempfile.mkdtemp(prefix='glow_test_'))
    config.folder = temp_folder
    config.cloud_config = None
    
    print("\n1. Loading experiment data...")
    config.prep_exp_orig()
    print(f"   ✓ Loaded exp_orig: {config.exp_orig.y.shape}")
    
    print("\n2. Running analysis (this will trigger memory profiling)...")
    
    # Wrap run_ana to capture exp and ana objects
    exp_captured = None
    ana_captured = None
    
    def run_ana_with_capture(config, **kwargs):
        nonlocal exp_captured, ana_captured
        exp, effect = config.get_exp_eff(**kwargs)
        exp_captured = exp

        for label, (Ana, ana_kwargs) in config.runner.iter_ana_kwargs():
            ana = Ana(exp=exp, **ana_kwargs)
            ana_captured = ana
            raise MemoryError("Simulated memory error for testing")
    
    try:
        run_ana_with_capture(config, seed=0, effect_llr=0.05)
    except MemoryError as e:
        print(f"\n3. Memory error caught: {e}")
        print("\n4. Generating memory profile...")
        print_memory_profile(config=config, exp=exp_captured, ana=ana_captured)
    
    # Also test with just config and exp (no ana)
    print("\n" + "=" * 60)
    print("Testing with config and exp only (no ana)")
    print("=" * 60)
    print_memory_profile(config=config, exp=exp_captured, ana=None)
    
    # Cleanup
    import shutil
    if temp_folder.exists():
        shutil.rmtree(temp_folder, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("✓ Memory profiling test complete")
    print("=" * 60)



if __name__ == '__main__':
    bench_memory_profile()
