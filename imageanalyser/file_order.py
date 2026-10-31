"""File creation timestamps, including Linux statx birth time when available."""
import ctypes
import os
import struct
import sys


def _load_statx():
    if not sys.platform.startswith('linux'):
        return None
    try:
        function = ctypes.CDLL(None, use_errno=True).statx
    except (OSError, AttributeError):
        return None
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


_statx = _load_statx()


def creation_time_ns(path, metadata):
    """Return (timestamp, used_fallback); ctime is not a creation date on Linux."""
    if hasattr(metadata, 'st_birthtime_ns'):
        return metadata.st_birthtime_ns, False
    if hasattr(metadata, 'st_birthtime'):
        return int(metadata.st_birthtime * 1_000_000_000), False
    if _statx is not None:
        # Linux's 256-byte statx ABI: mask at offset 0, birth timestamp at 0x50.
        buffer = ctypes.create_string_buffer(256)
        if _statx(-100, os.fsencode(path), 0x100, 0x800, buffer) == 0:
            mask = struct.unpack_from('=I', buffer, 0)[0]
            if mask & 0x800:  # STATX_BTIME is available on this filesystem.
                seconds, nanoseconds = struct.unpack_from('=qI', buffer, 0x50)
                return seconds * 1_000_000_000 + nanoseconds, False
    return metadata.st_mtime_ns, True
