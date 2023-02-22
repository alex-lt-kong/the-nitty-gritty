from distutils.core import setup, Extension


my_module = Extension('_mylib', sources=['mylib_wrap.cxx', 'mylib.cpp'])

setup (
    name = 'mylib',
    version = '0.1',
    author      = "SWIG Docs",
    description = """Simple swig example from docs""",
    ext_modules = [my_module],
    py_modules = ["mylib"],
)