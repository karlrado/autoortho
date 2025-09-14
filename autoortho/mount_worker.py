"""Isolated FUSE mount worker for macOS.

Launched as: python -m autoortho.mount_worker <root> <mountpoint> [--nothreads]
"""

from .autoortho_fuse import main_worker


def main():
    main_worker()


if __name__ == "__main__":
    main()


