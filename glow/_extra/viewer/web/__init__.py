"""Web demo layer for glow._extra.viewer.

Serves the viewer over HTTP: a Flask landing page mounting one baked
bundle per request (server.py), the bundle baker (bake_demos.py), and a
local bundle validator (play.py).

MANIFEST_NAME lives here rather than in bake_demos so the server can read
a baked set without importing the baker, which pulls in the benchmark
subpackage the deployed image leaves out (see deploy_hf.sh).
"""

MANIFEST_NAME = 'manifest.json'

# Sources whose data use terms let the derived maps be shared only with
# recipients bound by those same terms, which an anonymous visitor is
# not. Two things follow, at different points: bake_demos strips the
# real subject identifiers out of the bundle, and server withholds the
# per-subject images from the viewer it mounts. The statistics are
# unaffected by either.
GATED_SOURCES = frozenset({'hcp'})
