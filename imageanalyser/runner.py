"""Supervise the web process so UI restarts work without an external service manager."""
import signal
import subprocess
import sys

RESTART_EXIT_CODE = 75


def supervise(config_path):
    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while not stopping:
            child = subprocess.Popen(
                [sys.executable, '-m', 'imageanalyser.app', '--config', str(config_path), '--worker'],
                start_new_session=True,
            )
            if stopping:
                child.terminate()
            result = child.wait()
            if stopping:
                return 0
            if result != RESTART_EXIT_CODE:
                return result if result >= 0 else 128 - result
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
