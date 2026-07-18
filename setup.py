from setuptools import setup, Extension

lib = Extension('libsmerfs', 
                sources = ['src/smerfs.c', 'src/linalg.c', 'src/cov.c', 'src/ziggurat.c'],
                include_dirs=['src'],
                extra_compile_args=['-std=c99'])

setup(packages = ['smerfs', 'smerfs.tests'],
      ext_modules = [lib])
