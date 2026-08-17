"""Optional, non-core components bundled with glow.

The core library is the package root (analysis, effect, experiment and the
top-level modules graph / mask / plot). Everything here is peripheral tooling
built on top of that core and is not required to use it:

  - viewer    -- interactive result viewer (Dash/Plotly app).
  - benchmark -- the paper benchmark harness (sweeps, recorder, plots).

The leading underscore marks the whole subpackage as peripheral, not part of
the core public surface (and floats it to the top of the package listing), so
the core stands on its own at the package root.
"""
