from distutils.core import setup, Extension


my_module = Extension('_mylibs', sources=['mylibs_wrap.cxx', 'mylibs.cpp'])

setup (
    name = 'mylibs',
    version = '0.1',
    author      = "SWIG Docs",
    description = """Simple swig example from docs""",
    ext_modules = [my_module],
    py_modules = ["mylibs"],
)