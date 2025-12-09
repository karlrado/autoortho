#!/usr/bin/env python3
"""
Build script for AutoOrtho using PyInstaller.

Usage:
    python build_pyinstaller.py [--onefile] [--debug]

Options:
    --onefile   Create a single executable (slower startup, easier distribution)
    --debug     Include debug symbols and console output
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def check_dependencies():
    """Check that required build tools are installed."""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("✗ PyInstaller not found. Install with: pip install pyinstaller")
        return False
    
    # Check that autoortho directory exists
    if not os.path.isdir('autoortho'):
        print("✗ autoortho directory not found. Run from project root.")
        return False
    print("✓ autoortho directory found")
    
    return True


def clean_build():
    """Remove previous build artifacts."""
    dirs_to_clean = ['build', 'dist']
    for d in dirs_to_clean:
        if os.path.isdir(d):
            print(f"Cleaning {d}/...")
            shutil.rmtree(d)
    
    # Remove .pyc files
    for pyc in Path('.').rglob('*.pyc'):
        pyc.unlink()
    for pycache in Path('.').rglob('__pycache__'):
        if pycache.is_dir():
            shutil.rmtree(pycache)
    
    print("✓ Build directory cleaned")


def build(onefile=False, debug=False):
    """Run PyInstaller build."""
    cmd = ['pyinstaller']
    
    if onefile:
        # Modify spec for onefile mode
        cmd.extend(['--onefile'])
        cmd.extend(['--name', 'autoortho'])
        cmd.extend(['autoortho/__main__.py'])
        
        # Add all the binaries and data manually for onefile
        # This is more complex, so we recommend using the spec file
        print("Note: --onefile mode may require manual adjustment of binaries")
    else:
        # Use the spec file (recommended)
        cmd.append('autoortho.spec')
    
    if debug:
        cmd.append('--debug=all')
    
    cmd.append('--noconfirm')  # Don't ask for confirmation
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode == 0


def verify_build():
    """Verify the build output."""
    if sys.platform == 'win32':
        exe_path = Path('dist/autoortho/autoortho.exe')
    elif sys.platform == 'darwin':
        exe_path = Path('dist/autoortho/autoortho')
        app_path = Path('dist/AutoOrtho.app')
    else:
        exe_path = Path('dist/autoortho/autoortho')
    
    if exe_path.exists():
        print(f"✓ Executable created: {exe_path}")
        print(f"  Size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    else:
        print(f"✗ Executable not found at {exe_path}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build AutoOrtho with PyInstaller')
    parser.add_argument('--onefile', action='store_true', 
                        help='Create single executable')
    parser.add_argument('--debug', action='store_true',
                        help='Include debug symbols')
    parser.add_argument('--no-clean', action='store_true',
                        help='Skip cleaning previous builds')
    args = parser.parse_args()
    
    print("=" * 60)
    print("AutoOrtho PyInstaller Build")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Clean previous builds
    if not args.no_clean:
        clean_build()
    
    print()
    print("Building...")
    print("-" * 40)
    
    # Build
    if not build(onefile=args.onefile, debug=args.debug):
        print()
        print("✗ Build failed!")
        sys.exit(1)
    
    print()
    print("-" * 40)
    
    # Verify
    if verify_build():
        print()
        print("=" * 60)
        print("✓ Build completed successfully!")
        print("=" * 60)
        print()
        print("Output location: dist/autoortho/")
        if sys.platform == 'darwin':
            print("macOS App Bundle: dist/AutoOrtho.app/")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

