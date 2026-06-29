"""glow:viewer -- interactive dashboard for hierarchical segmentations.

Usage:
    from glow._extra.viewer import launch
    launch(ana_glow, exp, mask_target=mask_target)
"""

from .app import launch

__all__ = ['launch']
