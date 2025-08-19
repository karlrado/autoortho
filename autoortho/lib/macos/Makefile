CC=clang
CFLAGS=-O2 -Wall -fPIC -fvisibility=hidden
LDFLAGS=-dynamiclib -undefined dynamic_lookup

all: lib_stb_dxt.dylib libispc_texcomp.dylib

lib_stb_dxt.dylib: stb_dxt.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

libispc_texcomp.dylib: ispc_texcomp.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f *.dylib

.PHONY: all clean
