"""glow:viewer -- interactive dashboard for hierarchical segmentation results.

Usage::

    from glow.viewer import launch
    launch(ana_glow, mask_target=mask_target)
"""

from .app import launch

__all__ = ['launch']
