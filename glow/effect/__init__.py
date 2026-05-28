"""Synthetic effects: plant estimated effects into imaging data.

This subpackage models synthetic effects planted into imaging data: their
spatial extent (extent.py), the offset that imposes a target effect size
(impose.py), and the effect objects that tie support to a MANCOVA
decomposition (eff.py).
"""

from .eff import *
from .extent import *
from .impose import *
