"""Web demo layer for glow._extra.viewer.

Serves the viewer over HTTP: a Flask landing page mounting one baked
bundle per request (server.py), the bundle baker (bake_demos.py), and a
local bundle validator (play.py).

MANIFEST_NAME lives here rather than in bake_demos so the server can read
a baked set without importing the baker, which pulls in the benchmark
subpackage the deployed image leaves out (see deploy_hf.sh).
"""

MANIFEST_NAME = 'manifest.json'

# Sources whose data use terms bind whoever receives the data. A
# visitor accepts those terms before any bundle from one of these opens
# (server's /terms screen), and nothing behind that screen is withheld:
# the acceptance is the condition the terms set, so a viewer past it
# serves the individual images like any other.
GATED_SOURCES = frozenset({'hcp'})
