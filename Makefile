# -fPIC process independent code (for shared libraries)
# -Wall to warn about all my bugs :-)
# -std=c99 because I define ints in my loops and use complex numbers
# -O2 gives almost all the benefit of higher
CC = gcc -Wall -fPIC -shared -std=c99 -O2
SRC = src/*.c
LIB = build/libsmerfs.so

all: build ${LIB}
build:
	mkdir -p build

${LIB}: ${SRC} Makefile
	${CC} -o ${LIB} ${SRC}
clean:
	rm -rf build

