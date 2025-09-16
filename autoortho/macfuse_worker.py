import argparse
import logging
import os
from mfusepy import FUSE

from autoortho_fuse import AutoOrtho, fuse_option_profiles_by_os

log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mountpoint", required=True)
    ap.add_argument("--nothreads", action="store_true")
    ap.add_argument("--volname")
    args = ap.parse_args()

    log.info(f"MOUNT: {args.mountpoint}")
    additional_args = fuse_option_profiles_by_os(args.nothreads, args.volname)

    log.info(f"Starting FUSE mount")
    log.debug(f"Loading FUSE with options: "
            f"{', '.join(sorted(map(str, additional_args.keys())))}")

    try:
        FUSE(AutoOrtho(args.root), os.path.abspath(args.mountpoint), **additional_args)
        log.info(f"FUSE: Exiting mount {args.mountpoint}")
        return
    except Exception as e:
        log.error(f"FUSE mount failed with non-negotiable error: {e}")
        raise


if __name__ == "__main__":
    main()
