# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AutoOrtho.

This spec file bundles the application with all required native libraries
and executables for Windows, Linux, and macOS.

Usage:
    pyinstaller autoortho.spec

The resulting executable will be in dist/autoortho/
"""

import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all

block_cipher = None

# =============================================================================
# Collect modules with native extensions (required for Linux - have .so files)
# =============================================================================

def safe_collect_all(module_name):
    """Safely collect all components of a module, returning empty lists on failure."""
    try:
        return collect_all(module_name)
    except Exception:
        return [], [], []

def collect_stdlib_extension(module_name):
    """
    Collect a standard library C extension module (like _zstd in Python 3.14+).
    
    Unlike collect_all(), this works for standalone C extension modules that are
    part of Python's standard library and don't have an __init__.py.
    
    Returns: (datas, binaries, hiddenimports)
    """
    import importlib.util
    import sysconfig
    
    datas = []
    binaries = []
    hiddenimports = [module_name]
    
    try:
        # Find the module spec to get the actual .so/.pyd file location
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin and spec.origin != 'built-in':
            # It's a C extension with a file location
            module_path = spec.origin
            if os.path.isfile(module_path):
                # Get the filename (e.g., _zstd.cpython-314-x86_64-linux-gnu.so)
                module_file = os.path.basename(module_path)
                # Bundle it to the root of the package (PyInstaller will handle it)
                binaries.append((module_path, '.'))
                print(f"Found stdlib extension {module_name}: {module_path}")
        else:
            # Try finding it in the Python lib directory directly
            ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
            lib_dir = sysconfig.get_path('stdlib')
            
            # Check both lib-dynload and stdlib directories
            for search_dir in [lib_dir, os.path.join(lib_dir, 'lib-dynload')]:
                if search_dir and os.path.isdir(search_dir):
                    for f in os.listdir(search_dir):
                        if f.startswith(module_name) and f.endswith(ext_suffix):
                            full_path = os.path.join(search_dir, f)
                            binaries.append((full_path, '.'))
                            print(f"Found stdlib extension {module_name}: {full_path}")
                            break
    except Exception as e:
        print(f"Warning: Could not collect stdlib extension {module_name}: {e}")
    
    return datas, binaries, hiddenimports

# System monitoring
psutil_datas, psutil_binaries, psutil_hiddenimports = safe_collect_all('psutil')

# py7zr compression backends - these have native C extensions
# _zstd is Python 3.14+ standard library zstd compression
# NOTE: collect_all() doesn't work for stdlib C extensions, use collect_stdlib_extension()
zstd_datas, zstd_binaries, zstd_hiddenimports = collect_stdlib_extension('_zstd')
pyzstd_datas, pyzstd_binaries, pyzstd_hiddenimports = safe_collect_all('pyzstd')
# py7zr's compression package (contains zstd, ppmd, etc. wrappers)
compression_datas, compression_binaries, compression_hiddenimports = safe_collect_all('compression')
pybcj_datas, pybcj_binaries, pybcj_hiddenimports = safe_collect_all('pybcj')
pyppmd_datas, pyppmd_binaries, pyppmd_hiddenimports = safe_collect_all('pyppmd')
inflate64_datas, inflate64_binaries, inflate64_hiddenimports = safe_collect_all('inflate64')
brotli_datas, brotli_binaries, brotli_hiddenimports = safe_collect_all('brotli')

# Numerical/scientific
numpy_datas, numpy_binaries, numpy_hiddenimports = safe_collect_all('numpy')

# Async/networking (gevent stack)
greenlet_datas, greenlet_binaries, greenlet_hiddenimports = safe_collect_all('greenlet')
gevent_datas, gevent_binaries, gevent_hiddenimports = safe_collect_all('gevent')
zope_datas, zope_binaries, zope_hiddenimports = safe_collect_all('zope.interface')

# Serialization
msgpack_datas, msgpack_binaries, msgpack_hiddenimports = safe_collect_all('msgpack')

# Cryptography (used by py7zr)
crypto_datas, crypto_binaries, crypto_hiddenimports = safe_collect_all('Cryptodome')

# Flask dependencies with C extensions
markupsafe_datas, markupsafe_binaries, markupsafe_hiddenimports = safe_collect_all('markupsafe')

# Character encoding
charset_datas, charset_binaries, charset_hiddenimports = safe_collect_all('charset_normalizer')

# Collect all datas/binaries/hiddenimports for native modules
native_module_datas = (
    psutil_datas + zstd_datas + pyzstd_datas + compression_datas + pybcj_datas + pyppmd_datas + 
    inflate64_datas + brotli_datas + numpy_datas + greenlet_datas + gevent_datas + 
    zope_datas + msgpack_datas + crypto_datas + markupsafe_datas + charset_datas
)
native_module_binaries = (
    psutil_binaries + zstd_binaries + pyzstd_binaries + compression_binaries + pybcj_binaries + pyppmd_binaries + 
    inflate64_binaries + brotli_binaries + numpy_binaries + greenlet_binaries + gevent_binaries + 
    zope_binaries + msgpack_binaries + crypto_binaries + markupsafe_binaries + charset_binaries
)
native_module_hiddenimports = (
    psutil_hiddenimports + zstd_hiddenimports + pyzstd_hiddenimports + compression_hiddenimports + pybcj_hiddenimports + 
    pyppmd_hiddenimports + inflate64_hiddenimports + brotli_hiddenimports + numpy_hiddenimports + 
    greenlet_hiddenimports + gevent_hiddenimports + zope_hiddenimports + msgpack_hiddenimports + 
    crypto_hiddenimports + markupsafe_hiddenimports + charset_hiddenimports
)

# Determine platform
if sys.platform == 'win32':
    platform_name = 'windows'
    exe_suffix = '.exe'
    dll_suffix = '.dll'
elif sys.platform == 'darwin':
    platform_name = 'macos'
    exe_suffix = ''
    dll_suffix = '.dylib'
else:
    platform_name = 'linux'
    exe_suffix = ''
    dll_suffix = '.so'

# Base paths
autoortho_path = os.path.join(os.getcwd(), 'autoortho')
lib_path = os.path.join(autoortho_path, 'lib', platform_name)
aoimage_path = os.path.join(autoortho_path, 'aoimage')
aopipeline_path = os.path.join(autoortho_path, 'aopipeline')
aopipeline_lib_path = os.path.join(aopipeline_path, 'lib', platform_name)

# =============================================================================
# Binary files (DLLs, shared libraries, executables)
# =============================================================================
binaries = []

def find_system_library(lib_name):
    """
    Find a system shared library by name.
    Returns the full path if found, None otherwise.
    """
    import subprocess
    import ctypes.util
    
    # Try ctypes.util.find_library first
    lib_path = ctypes.util.find_library(lib_name)
    if lib_path:
        # find_library returns just the name on Linux, need to resolve full path
        if not os.path.isabs(lib_path):
            # Use ldconfig or search common paths
            try:
                result = subprocess.run(
                    ['ldconfig', '-p'],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.splitlines():
                    if lib_name in line and '=>' in line:
                        path = line.split('=>')[1].strip().split()[0]
                        if os.path.isfile(path):
                            return path
            except Exception:
                pass
            
            # Search common library paths
            search_paths = [
                '/lib/x86_64-linux-gnu',
                '/usr/lib/x86_64-linux-gnu', 
                '/lib64',
                '/usr/lib64',
                '/usr/local/lib',
            ]
            for search_path in search_paths:
                for f in os.listdir(search_path) if os.path.isdir(search_path) else []:
                    if f.startswith(f'lib{lib_name}.so'):
                        full_path = os.path.join(search_path, f)
                        if os.path.isfile(full_path):
                            return full_path
        else:
            return lib_path
    return None

# Bundle libexpat for Linux (required by pyexpat, version-sensitive)
if sys.platform.startswith('linux'):
    expat_lib = find_system_library('expat')
    if expat_lib:
        binaries.append((expat_lib, '.'))
        print(f"Found libexpat: {expat_lib}")
        # Also get the unversioned symlink target if it's a symlink
        if os.path.islink(expat_lib):
            real_path = os.path.realpath(expat_lib)
            if real_path != expat_lib and os.path.isfile(real_path):
                binaries.append((real_path, '.'))
                print(f"Found libexpat (real): {real_path}")
    else:
        print("WARNING: libexpat not found - pyexpat may fail at runtime")

# AoImage library
if sys.platform == 'win32':
    binaries.append((os.path.join(aoimage_path, 'aoimage.dll'), 'autoortho/aoimage'))
elif sys.platform == 'darwin':
    binaries.append((os.path.join(aoimage_path, 'aoimage.dylib'), 'autoortho/aoimage'))
else:
    binaries.append((os.path.join(aoimage_path, 'aoimage.so'), 'autoortho/aoimage'))

# AoPipeline library and dependencies
if sys.platform == 'win32':
    binaries.append((os.path.join(aopipeline_lib_path, 'aopipeline.dll'), 'autoortho/aopipeline/lib/windows'))
    binaries.append((os.path.join(aopipeline_lib_path, 'libgcc_s_seh-1.dll'), 'autoortho/aopipeline/lib/windows'))
    binaries.append((os.path.join(aopipeline_lib_path, 'libgomp-1.dll'), 'autoortho/aopipeline/lib/windows'))
    binaries.append((os.path.join(aopipeline_lib_path, 'libturbojpeg.dll'), 'autoortho/aopipeline/lib/windows'))
    binaries.append((os.path.join(aopipeline_lib_path, 'libwinpthread-1.dll'), 'autoortho/aopipeline/lib/windows'))
elif sys.platform == 'darwin':
    binaries.append((os.path.join(aopipeline_lib_path, 'libaopipeline.dylib'), 'autoortho/aopipeline/lib/macos'))
else:
    binaries.append((os.path.join(aopipeline_lib_path, 'libaopipeline.so'), 'autoortho/aopipeline/lib/linux'))

# Platform-specific compression libraries and tools
if sys.platform == 'win32':
    # Compression libraries
    binaries.append((os.path.join(lib_path, 'ispc_texcomp.dll'), 'autoortho/lib/windows'))
    binaries.append((os.path.join(lib_path, 'stb_dxt.dll'), 'autoortho/lib/windows'))
    
    # 7-Zip
    binaries.append((os.path.join(lib_path, '7zip', '7za.exe'), 'autoortho/lib/windows/7zip'))
    binaries.append((os.path.join(lib_path, '7zip', '7za.dll'), 'autoortho/lib/windows/7zip'))
    binaries.append((os.path.join(lib_path, '7zip', '7zxa.dll'), 'autoortho/lib/windows/7zip'))
    
    # DSFTool
    binaries.append((os.path.join(lib_path, 'DSFTool.exe'), 'autoortho/lib/windows'))

elif sys.platform == 'darwin':
    # Compression libraries
    binaries.append((os.path.join(lib_path, 'libispc_texcomp.dylib'), 'autoortho/lib/macos'))
    binaries.append((os.path.join(lib_path, 'libstbdxt.dylib'), 'autoortho/lib/macos'))
    
    # 7-Zip
    binaries.append((os.path.join(lib_path, '7zip', '7zz'), 'autoortho/lib/macos/7zip'))
    
    # DSFTool
    binaries.append((os.path.join(lib_path, 'DSFTool'), 'autoortho/lib/macos'))

else:  # Linux
    # Compression libraries
    binaries.append((os.path.join(lib_path, 'libispc_texcomp.so'), 'autoortho/lib/linux'))
    binaries.append((os.path.join(lib_path, 'lib_stb_dxt.so'), 'autoortho/lib/linux'))
    
    # 7-Zip
    binaries.append((os.path.join(lib_path, '7zip', '7zz'), 'autoortho/lib/linux/7zip'))
    
    # DSFTool
    binaries.append((os.path.join(lib_path, 'DSFTool'), 'autoortho/lib/linux'))

# =============================================================================
# Data files
# =============================================================================
datas = []

# Version file - auto-generate from git if not present
version_file = os.path.join(autoortho_path, '.version')
if not os.path.exists(version_file):
    # Try to get version from git
    import subprocess
    try:
        git_version = subprocess.check_output(
            ['git', 'describe', '--tags', '--always', '--dirty'],
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd()
        ).decode().strip()
        # Write to .version file
        with open(version_file, 'w') as f:
            f.write(git_version)
        print(f"Generated .version file: {git_version}")
    except Exception as e:
        # Fallback: use current date/time as version
        from datetime import datetime
        fallback_version = f"build-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with open(version_file, 'w') as f:
            f.write(fallback_version)
        print(f"Generated fallback .version file: {fallback_version}")

if os.path.exists(version_file):
    datas.append((version_file, 'autoortho'))

# Flask templates (required for web UI)
templates_path = os.path.join(autoortho_path, 'templates')
if os.path.isdir(templates_path):
    datas.append((templates_path, 'autoortho/templates'))

# Images/icons for UI
imgs_path = os.path.join(autoortho_path, 'imgs')
if os.path.isdir(imgs_path):
    datas.append((imgs_path, 'autoortho/imgs'))

# 7-Zip license
if sys.platform == 'win32':
    license_path = os.path.join(lib_path, '7zip', 'License.txt')
else:
    license_path = os.path.join(lib_path, '7zip', 'License.txt')
if os.path.exists(license_path):
    datas.append((license_path, f'autoortho/lib/{platform_name}/7zip'))

# SSL Certificates (required for macOS - system certs not accessible in bundled app)
if sys.platform == 'darwin':
    import certifi
    cacert_path = certifi.where()
    if os.path.exists(cacert_path):
        # Bundle certifi's CA bundle for SSL verification
        datas.append((cacert_path, 'certifi'))
        print(f"Bundling SSL certificates from: {cacert_path}")

# =============================================================================
# Hidden imports (modules that PyInstaller might miss)
# =============================================================================
hiddenimports = [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'geocoder',
    'geocoder.osm',
    'psutil',
    'requests',
    'PIL',
    'PIL.Image',
    # Flask-SocketIO async drivers (required for frozen apps)
    'engineio.async_drivers.threading',
    'socketio.async_drivers.threading',
    # AoPipeline modules (lazy imports - PyInstaller won't detect automatically)
    'autoortho.aopipeline',
    'autoortho.aopipeline.AoCache',
    'autoortho.aopipeline.AoDecode',
    'autoortho.aopipeline.AoDDS',
    'autoortho.aopipeline.AoBundle',
    'autoortho.aopipeline.AoBundle2',
    'autoortho.aopipeline.bundle_consolidator',
    'autoortho.aopipeline.fallback_resolver',
    # py7zr compression backends (have native C extensions)
    'py7zr',
    'py7zr.compressor',
    'py7zr.archiveinfo',
    'py7zr.io',  # Used by aoseasons.py for BytesIOFactory
    '_zstd',  # Python 3.14+ standard library zstd
    'compression',  # py7zr's compression package
    'compression.zstd',  # py7zr's zstd wrapper (imports _zstd)
    'pyzstd',  # fallback if _zstd unavailable (optional)
    'pybcj',
    'pyppmd',
    'inflate64',
    'multivolumefile',
    'brotli',
    # Cryptography (used by py7zr for encrypted archives)
    'Cryptodome',
    'Cryptodome.Cipher',
    'Cryptodome.Cipher.AES',
    'Cryptodome.Hash',
    'Cryptodome.Random',
    # Async/networking stack
    'greenlet',
    'gevent',
    'gevent.resolver',
    'gevent.ssl',
    'zope',
    'zope.interface',
    'zope.event',
    # Serialization
    'msgpack',
    # NumPy
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',
    # Flask/Jinja dependencies
    'markupsafe',
    'markupsafe._speedups',
    # Character encoding
    'charset_normalizer',
    'charset_normalizer.md',
]

# Platform-specific hidden imports
if sys.platform == 'darwin':
    hiddenimports.extend([
        'macsetup',
        'macfuse_worker',
    ])
elif sys.platform == 'win32':
    hiddenimports.extend([
        'winsetup',
        'winfspy',
        'winfspy.plumbing',
    ])

# =============================================================================
# Analysis
# =============================================================================
a = Analysis(
    [os.path.join(autoortho_path, '__main__.py')],
    pathex=[autoortho_path],
    binaries=binaries + native_module_binaries,
    datas=datas + native_module_datas,
    hiddenimports=hiddenimports + native_module_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'numpy.testing',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# =============================================================================
# PYZ (Python bytecode archive)
# =============================================================================
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# =============================================================================
# EXE
# =============================================================================
# Determine icon path based on platform
if sys.platform == 'win32':
    icon_path = os.path.join(autoortho_path, 'imgs', 'ao-icon.ico')
elif sys.platform == 'darwin':
    icon_path = os.path.join(autoortho_path, 'imgs', 'ao-icon.icns')
else:
    icon_path = None

# Verify icon exists
if icon_path and os.path.exists(icon_path):
    print(f"Using icon: {icon_path}")
else:
    print(f"WARNING: Icon not found at {icon_path}")
    icon_path = None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='autoortho',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI-only mode - no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
    contents_directory='ao_files',  # Custom folder name instead of '_internal'
)

# =============================================================================
# COLLECT (bundle everything together)
# =============================================================================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='autoortho',
)

# =============================================================================
# macOS App Bundle (optional)
# =============================================================================
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='AutoOrtho.app',
        icon=os.path.join(autoortho_path, 'imgs', 'ao-icon.icns'),
        bundle_identifier='com.autoortho.app',
        info_plist={
            'CFBundleName': 'AutoOrtho',
            'CFBundleDisplayName': 'AutoOrtho',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.15',
        },
    )

