from distutils.core import setup, Extension
setup(name='mylib', version='1.0', ext_modules=[Extension('mylib', ['mylib.c'])])