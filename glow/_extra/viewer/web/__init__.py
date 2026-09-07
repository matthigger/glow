"""Web demo layer for glow._extra.viewer.

Serves the viewer over HTTP: a Flask landing page mounting one baked
bundle per request (server.py), the bundle baker (bake_demos.py), and a
local bundle validator (play.py).

MANIFEST_NAME lives here rather than in bake_demos so the server can read
a baked set without importing the baker, which pulls in the benchmark
subpackage the deployed image leaves out (see deploy_hf.sh).
"""

MANIFEST_NAME = 'manifest.json'
