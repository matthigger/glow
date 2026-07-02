"""OS port and process management for the glow viewer server.

Helpers to test whether the requested TCP port is free and, with the
user's confirmation, free it by terminating the process holding it.
Cross-platform (POSIX lsof/kill, Windows netstat/taskkill); no Dash
coupling.
"""

import os
import signal
import socket
import subprocess
import sys
import time


def _check_port(port):
    """Check whether port is available; if not, offer to free it.

    Uses a plain socket bind test (cross-platform). If the port is
    occupied, prompts for confirmation before killing the blocking process.

    Args:
        port (int): TCP port to probe on 127.0.0.1.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('127.0.0.1', port))
        sock.close()
        return
    except OSError:
        pass

    print(f'\n  Port {port} is already in use.')
    answer = input('  Kill the process using it? [y/N] ').strip().lower()
    if answer not in ('y', 'yes'):
        print('  Aborted.  Use --port to choose a different port.')
        raise SystemExit(1)

    pids = _find_pids_on_port(port)
    if not pids:
        print(f'  Could not identify process on port {port}.  '
              f'Use --port to choose a different port.')
        raise SystemExit(1)

    if sys.platform == 'win32':
        for pid in pids:
            subprocess.call(
                ['taskkill', '/F', '/PID', str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    else:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        # give SIGTERM 2 seconds to work before escalating to SIGKILL
        if not _wait_for_port(port, socket, timeout=2.0):
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    if _wait_for_port(port, socket, timeout=5.0):
        print(f'  Freed port {port}.')
    else:
        print(f'  Port {port} still in use.  Use --port to choose '
              f'a different port.')
        raise SystemExit(1)


def _find_pids_on_port(port):
    """Return PIDs listening on port (best-effort, cross-platform)."""
    pids = []
    if sys.platform == 'win32':
        try:
            out = subprocess.check_output(
                ['netstat', '-ano'], stderr=subprocess.DEVNULL,
            ).decode()
            for line in out.splitlines():
                if f':{port}' in line and 'LISTENING' in line:
                    try:
                        pids.append(int(line.strip().split()[-1]))
                    except ValueError:
                        pass
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    else:
        try:
            out = subprocess.check_output(
                ['lsof', '-t', '-i', f':{port}'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            for pid_str in out.splitlines():
                try:
                    pids.append(int(pid_str))
                except ValueError:
                    pass
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return pids


def _wait_for_port(port, socket_mod, timeout=5.0):
    """Poll until port is free; return True if freed within timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
        s.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
        try:
            s.bind(('127.0.0.1', port))
            s.close()
            return True
        except OSError:
            s.close()
        time.sleep(0.1)
    return False
