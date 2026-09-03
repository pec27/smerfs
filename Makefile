#lib = Extension('libsmerfs',
#                sources = ['src/smerfs.c', 'src/linalg.c', 'src/cov.c', 'src/ziggurat.c'],
#                include_dirs=['src'],
#                extra_compile_args=['-std=c99'])

# Find the cpython suffix for the library, e.g. ".cpython-39-darwin.so"
CPYTHON_SUFFIX := $(shell python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

LIB = libsmerfs$(CPYTHON_SUFFIX)
# -Weverything -Wsign-conversion -Wcovered-switch-default -Wshorten-64-to-32
CC = cc -Wextra -Wunused-variable -Wunused-function -Wsign-conversion -Wunused-but-set-variable -Wpedantic -Wcast-align -Wswitch-enum  -Wshadow -Wimplicit-fallthrough -Werror -fPIC -shared -std=c2x
#-O3

HDR = src/zig.h

# Absolute sources (not auto-generated)
SRC_ABS = src/smerfs.c src/linalg.c src/cov.c src/ziggurat.c

LIB_DEPENDS = ${SRC_ABS} ${HDR} Makefile

$(LIB): ${LIB_DEPENDS}
	${CC} -o ${LIB} ${SRC_ABS}

all: ${LIB}

clean:
	rm -rf build

