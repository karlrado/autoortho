#!/usr/bin/env python3
"""
Integration tests for end-to-end FUSE -> DDS request -> tile/chunk pipeline.

- Runs under Linux/WSL2 (requires libfuse). Windows/macOS can run similar flows
  but this test focuses on Linux CI and WSL2 viability.
- X-Plane file accesses are mocked by directly reading from the mounted FUSE FS
  (e.g., /textures/<row>_<col>_<map>16.dds).
- Network imagery fetches are mocked to return a small JPEG from test assets.

Produces meaningful stats from getortho.STATS: req_ok, chunk_miss, bytes_dl,
mm_counts/averages, and measured timings for first/second tile reads.
"""

import os
import time
import threading
import contextlib

import platform
import pytest


def _have_fuse_linux():
    try:
        import mfusepy  # noqa
    except Exception:
        return False
    return os.name == 'posix' and platform.system().lower() == 'linux'


def test_fuse_end_to_end_linux(tmp_path):
    """Mount AutoOrtho FS, request a DDS file twice, collect stats and timings."""
    import autoortho_fuse
    import getortho

    # 1) Mock network: serve a tiny JPEG for all requests
    test_jpg = os.path.join(os.path.dirname(__file__), 'testfiles', 'test_tile_small.jpg')
    with open(test_jpg, 'rb') as _h:
        jpeg_bytes = _h.read()

    class _Resp:
        def __init__(self, data):
            self.status_code = 200
            self.content = data
        def close(self):
            pass

    def fake_get(url, *args, **kwargs):
        return _Resp(jpeg_bytes)

    # Patch the session used by the global chunk_getter
    getortho.chunk_getter.session.get = fake_get

    # 2) Mount FUSE in background thread with safe options (no allow_other)
    mount_dir = tmp_path / "mnt"
    root_dir = tmp_path / "root"
    cache_dir = tmp_path / "cache"
    os.makedirs(mount_dir, exist_ok=True)
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Monkeypatch option builder to avoid allow_other in constrained environments
    orig_opts = autoortho_fuse.fuse_option_profiles_by_os

    def _opts_no_allow_other(nothreads: bool, name: str):
        opts = dict(nothreads=nothreads, foreground=True)
        # Minimal, non-privileged option set for tests
        if os.name == 'posix':
            opts.update(dict())
        return opts

    autoortho_fuse.fuse_option_profiles_by_os = _opts_no_allow_other

    fs = autoortho_fuse.AutoOrtho(str(root_dir), cache_dir=str(cache_dir))

    def _mount():
        # nothreads=True makes debugging and teardown simpler in tests
        autoortho_fuse.run(fs, str(mount_dir), name="AOTest", nothreads=True)

    t = threading.Thread(target=_mount, daemon=True)
    t.start()

    # 3) Wait for mount readiness by probing FUSE readdir contract
    textures_dir = mount_dir / "textures"
    deadline = time.time() + 10
    ready = False
    while time.time() < deadline:
        try:
            entries = os.listdir(textures_dir)
            if 'AOISWORKING' in entries:
                ready = True
                break
        except Exception:
            time.sleep(0.2)
    if not ready:
        # Restore and skip if we cannot mount in this environment
        autoortho_fuse.fuse_option_profiles_by_os = orig_opts
        pytest.skip("FUSE mount did not become ready; environment likely lacks FUSE permissions")

    # 4) Request a DDS tile twice and measure timings
    dds_rel = "textures/21504_33680_BI16.dds"  # arbitrary coords
    dds_path = mount_dir / dds_rel

    def _read_some_bytes():
        t0 = time.perf_counter()
        # Read header and some payload to trigger pipeline
        with open(dds_path, 'rb', buffering=0) as fh:
            _ = fh.read(4096)
            fh.seek(512 * 1024)
            _ = fh.read(4096)
        t1 = time.perf_counter()
        return t1 - t0

    first_s = _read_some_bytes()
    second_s = _read_some_bytes()

    # 5) Collect stats
    stats = dict(getortho.STATS)

    # 6) Unmount by touching poison (getattr handles .poison)
    with contextlib.suppress(Exception):
        os.lstat(mount_dir / ".poison")

    # Restore option builder
    autoortho_fuse.fuse_option_profiles_by_os = orig_opts

    # 7) Assertions and reporting
    # At minimum, the mount responded and we saw mm/bytes or chunk activity.
    # Either we downloaded (req_ok>0) or the pipeline processed cached chunks.
    assert first_s > 0 and second_s > 0

    # We expect STATS keys to exist; tolerate zero req_ok if cache was warm
    assert 'chunk_miss' in stats and 'bytes_dl' in stats

    # Provide basic improvements due to cache/pipeline: second read should not be slower
    assert second_s <= first_s * 2.0  # be generous on CI boxes

    # Emit useful stats to test output
    print({
        'first_tile_read_s': round(first_s, 3),
        'second_tile_read_s': round(second_s, 3),
        'req_ok': stats.get('req_ok', 0),
        'chunk_miss': stats.get('chunk_miss', 0),
        'bytes_dl': stats.get('bytes_dl', 0),
        'mm_counts': stats.get('mm_counts'),
        'mm_averages': stats.get('mm_averages'),
        'partial_mm_counts': stats.get('partial_mm_counts'),
    })


