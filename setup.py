from setuptools import setup, Extension

lib = Extension('libsmerfs',
                sources = ['src/smerfs.c'])


setup(name='smerfs', version='0.1',
      author="Peter Creasey",
      author_email="pec27",
      description='Stochastic Markov Evaluation of Random Fields on the Sphere (SMERFS)',
      url="http://github.com/pec27/smerfs",
      package_dir = {'smerfs': 'smerfs'},
      packages = ['smerfs', 'smerfs.tests'],
      license='MIT',
      install_requires=['numpy', 'scipy'],
      ext_modules = [lib],
      test_suite='nose.collector',
      tests_require=['nose'])

