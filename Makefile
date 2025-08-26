ZIP?=zip
VERSION?=0.0.0
# Sanitize VERSION for use in filenames (replace any non-safe char with '-')
SAFE_VERSION:=$(shell echo "$(VERSION)" | sed -e 's/[^A-Za-z0-9._-]/-/g')

autoortho.pyz:
	mkdir -p build/autoortho
	cp -r autoortho/* build/autoortho/.
	python3 -m pip install -U -r ./build/autoortho/build-reqs.txt --target ./build/autoortho
	cd build && python3 -m zipapp -p "/usr/bin/env python3" autoortho

lin_bin: autoortho_lin_$(SAFE_VERSION).bin
autoortho_lin_$(SAFE_VERSION).bin: autoortho/*.py
	docker run --rm -v `pwd`:/code ubuntu:noble /bin/bash -c "cd /code; ./buildreqs.sh; . .venv/bin/activate; time make bin VERSION=$(VERSION)"
	mv autoortho_lin.bin $@

enter:
	docker run --rm -it -v `pwd`:/code ubuntu:focal /bin/bash

autoortho/.version:
	echo "$(VERSION)" > $@

bin: autoortho/.version
	python3 -m nuitka --verbose --verbose-output=nuitka.log \
		--linux-icon=autoortho/imgs/ao-icon.ico \
		--enable-plugin=pyside6 \
		--include-data-file=./autoortho/.version*=. \
		--include-data-file=./autoortho/templates/*.html=templates/ \
		--include-data-file=./autoortho/lib/linux/*.so=lib/linux/ \
		--include-data-file=./autoortho/aoimage/*.so=aoimage/ \
		--include-data-dir=./autoortho/imgs=imgs \
		--onefile \
		./autoortho/__main__.py -o autoortho_lin.bin

mac_app: AutoOrtho.app
AutoOrtho.app: autoortho/.version
	python3 -m nuitka --verbose --verbose-output=nuitka.log \
		--standalone \
		--macos-create-app-bundle \
		--macos-target-arch=arm64 \
		--macos-app-name=AutoOrtho \
		--macos-app-icon=autoortho/imgs/ao-icon.icns \
		--enable-plugin=pyside6 \
		--include-data-file=./autoortho/.version*=. \
		--include-data-file=./autoortho/templates/*.html=templates/ \
		--include-data-file=./autoortho/lib/macos/*.dylib=lib/macos/ \
		--include-data-file=./autoortho/aoimage/*.dylib=aoimage/ \
		--include-data-dir=./autoortho/imgs=imgs \
		./autoortho/__main__.py -o AutoOrtho.app

mac_zip: AutoOrtho_mac_$(SAFE_VERSION).zip
AutoOrtho_mac_$(SAFE_VERSION).zip: AutoOrtho.app
	$(ZIP) -r $@ AutoOrtho.app

_autoortho_win.exe: autoortho/.version
	python3 -m nuitka --verbose --verbose-output=nuitka.log \
		--mingw64 \
		--disable-ccache \
		--enable-plugin=pyside6 \
		--windows-icon-from-ico=autoortho/imgs/ao-icon.ico \
		--assume-yes-for-downloads \
		--include-data-file=./autoortho/.version*=./ \
		--include-data-file=./autoortho/templates/*.html=templates/ \
		--include-data-file=./autoortho/lib/windows/*=lib/windows/ \
		--include-data-file=./autoortho/aoimage/*.dll=aoimage/ \
		--include-data-dir=./autoortho/imgs=imgs \
		--onefile \
		--disable-console \
		./autoortho/__main__.py -o autoortho_win.exe

__main__.dist: autoortho/.version
	python3 -m nuitka --verbose --verbose-output=nuitka.log \
		--mingw64 \
		--disable-ccache \
		--enable-plugin=pyside6 \
		--windows-icon-from-ico=autoortho/imgs/ao-icon.ico \
		--assume-yes-for-downloads \
		--include-data-file=./autoortho/.version*=./ \
		--include-data-file=./autoortho/templates/*.html=templates/ \
		--include-data-file=./autoortho/lib/windows/*=lib/windows/ \
		--include-data-file=./autoortho/aoimage/*.dll=aoimage/ \
		--include-data-dir=./autoortho/imgs=imgs \
		--standalone \
		--disable-console \
		./autoortho/__main__.py -o autoortho_win.exe

win_exe: AutoOrtho_win_$(SAFE_VERSION).exe
AutoOrtho_win_$(SAFE_VERSION).exe: __main__.dist
	cp autoortho/imgs/ao-icon.ico .
	makensis -DPRODUCT_VERSION=$(VERSION) installer.nsi
	mv AutoOrtho.exe $@

win_zip: autoortho_win_$(SAFE_VERSION).zip
autoortho_win_$(SAFE_VERSION).zip: __main__.dist
	mv __main__.dist autoortho_release
	$(ZIP) $@ autoortho_release

testperf:
	python3 -m nuitka --verbose --verbose-output=nuitka.log  --include-data-dir=./autoortho/lib=lib --include-data-dir=./autoortho/testfiles=testfiles --onefile ./autoortho/perftest.py

%.txt: %.in
	pip-compile $<

serve_docs:
	docker run -p 8000:8000 -v `pwd`:/docs squidfunk/mkdocs-material

clean:
	-rm -rf build
	-rm -rf __main__.dist
	-rm -rf __main__.build
	
