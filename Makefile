ZIP?=zip
VERSION?=0.0.0
# Sanitize VERSION for use in filenames (replace any non-safe char with '-')
SAFE_VERSION:=$(shell echo "$(VERSION)" | sed -e 's/[^A-Za-z0-9._-]/-/g')
CODE_NAME:=$(shell grep UBUNTU_CODENAME /etc/os-release 2>/dev/null | cut -d= -f2 || echo "unknown")
.PHONY: mac_app clean
SHELL := /bin/bash
.ONESHELL:

# =============================================================================
# Version file
# =============================================================================
autoortho/.version:
	echo "$(VERSION)" > $@

# =============================================================================
# Linux Build (PyInstaller)
# =============================================================================
lin_bin: autoortho_lin_$(SAFE_VERSION)_${CODE_NAME}.bin
autoortho_lin_$(SAFE_VERSION)_${CODE_NAME}.bin: autoortho/*.py autoortho/.version
	docker run --rm -v `pwd`:/code ubuntu:${CODE_NAME} /bin/bash -c "cd /code; ./buildreqs.sh; . .venv/bin/activate; time make bin VERSION=$(VERSION)"
	mv autoortho_lin.bin $@

lin_tar: autoortho_linux_$(SAFE_VERSION)_${CODE_NAME}.tar.gz
autoortho_linux_$(SAFE_VERSION)_${CODE_NAME}.tar.gz: autoortho_lin_$(SAFE_VERSION)_${CODE_NAME}.bin
	chmod a+x $<
	tar -czf $@ $<

bin: autoortho/.version
	# Ensure required Linux libraries and helper binaries are executable before packaging
	chmod +x autoortho/lib/linux/7zip/7zz || true
	chmod +x autoortho/lib/linux/DSFTool || true
	chmod +x autoortho/lib/linux/*.so || true
	chmod +x autoortho/aoimage/aoimage.so || true
	# Build with PyInstaller
	.venv/bin/python3 -m PyInstaller \
		--noconfirm \
		--clean \
		--log-level INFO \
		autoortho.spec
	# Move output for compatibility with existing scripts
	mv dist/autoortho/autoortho autoortho_lin.bin || true

# =============================================================================
# macOS Build (PyInstaller)
# =============================================================================
mac_dist: autoortho/.version
	# Ensure required macOS libraries and helper binaries are executable before packaging
	chmod +x autoortho/lib/macos/7zip/7zz || true
	chmod +x autoortho/lib/macos/DSFTool || true
	chmod +x autoortho/lib/macos/*.dylib || true
	chmod +x autoortho/aoimage/aoimage.dylib || true
	# Build with PyInstaller
	python3 -m PyInstaller \
		--noconfirm \
		--clean \
		--log-level INFO \
		autoortho.spec

AutoOrtho.app: mac_dist
	# PyInstaller creates the app bundle automatically on macOS
	@if [ -d "dist/AutoOrtho.app" ]; then \
		rm -rf AutoOrtho.app; \
		mv dist/AutoOrtho.app AutoOrtho.app; \
	else \
		echo "Creating app bundle from dist/autoortho..."; \
		rm -rf AutoOrtho.app; \
		mkdir -p AutoOrtho.app/Contents/MacOS; \
		mkdir -p AutoOrtho.app/Contents/Resources; \
		cp -r dist/autoortho/* AutoOrtho.app/Contents/MacOS/; \
		echo '<?xml version="1.0" encoding="UTF-8"?>' > AutoOrtho.app/Contents/Info.plist; \
		echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' >> AutoOrtho.app/Contents/Info.plist; \
		echo '<plist version="1.0"><dict>' >> AutoOrtho.app/Contents/Info.plist; \
		echo '<key>CFBundleName</key><string>AutoOrtho</string>' >> AutoOrtho.app/Contents/Info.plist; \
		echo '<key>CFBundleExecutable</key><string>autoortho</string>' >> AutoOrtho.app/Contents/Info.plist; \
		echo '<key>CFBundleIdentifier</key><string>com.autoortho.app</string>' >> AutoOrtho.app/Contents/Info.plist; \
		echo '<key>CFBundleVersion</key><string>$(VERSION)</string>' >> AutoOrtho.app/Contents/Info.plist; \
		echo '<key>CFBundleShortVersionString</key><string>$(VERSION)</string>' >> AutoOrtho.app/Contents/Info.plist; \
		echo '<key>NSHighResolutionCapable</key><true/>' >> AutoOrtho.app/Contents/Info.plist; \
		echo '</dict></plist>' >> AutoOrtho.app/Contents/Info.plist; \
	fi

mac_app: AutoOrtho.app

AutoOrtho_mac_$(SAFE_VERSION).zip: AutoOrtho.app
	# Include the quarantine fix script alongside the app
	cp scripts/fix_macos_quarantine.command . 2>/dev/null || true
	chmod +x fix_macos_quarantine.command 2>/dev/null || true
	$(ZIP) -r $@ AutoOrtho.app fix_macos_quarantine.command
	rm -f fix_macos_quarantine.command
mac_zip: AutoOrtho_mac_$(SAFE_VERSION).zip

# =============================================================================
# Windows Build (PyInstaller)
# =============================================================================
win_dist: autoortho/.version
	# Build with PyInstaller (Windows)
	python3 -m PyInstaller \
		--noconfirm \
		--clean \
		--log-level INFO \
		autoortho.spec

__main__.dist: win_dist
	# Rename for compatibility with existing scripts
	@if [ -d "dist/autoortho" ]; then \
		rm -rf __main__.dist; \
		mv dist/autoortho __main__.dist; \
	fi

win_exe: AutoOrtho_win_$(SAFE_VERSION).exe
AutoOrtho_win_$(SAFE_VERSION).exe: __main__.dist
	cp autoortho/imgs/ao-icon.ico . || true
	makensis -DPRODUCT_VERSION=$(VERSION) installer.nsi
	mv AutoOrtho.exe $@

win_zip: autoortho_win_$(SAFE_VERSION).zip
autoortho_win_$(SAFE_VERSION).zip: __main__.dist
	mv __main__.dist autoortho_release
	$(ZIP) $@ autoortho_release

# =============================================================================
# Development helpers
# =============================================================================
enter:
	docker run --rm -it -v `pwd`:/code ubuntu:focal /bin/bash

autoortho.pyz:
	mkdir -p build/autoortho
	cp -r autoortho/* build/autoortho/.
	python3 -m pip install -U -r ./build/autoortho/build-reqs.txt --target ./build/autoortho
	cd build && python3 -m zipapp -p "/usr/bin/env python3" autoortho

%.txt: %.in
	pip-compile $<

serve_docs:
	docker run -p 8000:8000 -v `pwd`:/docs squidfunk/mkdocs-material

clean:
	rm -rf AutoOrtho.app *.zip *.tar.gz *.bin build dist __main__.dist autoortho_release
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
