# Python interface - SWIG

* This project is a continuation of its [C version](../../../c/04_fancy-functions/04_python-interface/3_calling-c-from-python-swig/)

* To generate SWIG files: `swig -c++ -python mylibs.i`

* To make it work on Windows is a bit more complicated:
    1. make sure a proper Python is installed. A "proper" Python should have
    at least an interpreter, an `include` directory and `libs` directory
    for SWIG to compile and link.
    1. Make sure SWIG is properly installed.
    1. Change the `include_directories()` and `link_directories()` in
    `CMakeLists.txt` to make sure build tools can link all the components
    together.

## References

* https://swig.org/Doc3.0/Python.html#Python_nn20
* https://stackoverflow.com/questions/8776328/swig-interfacing-c-library-to-python-creating-iterable-python-data-type-from/8828454#8828454
* https://stackoverflow.com/questions/12392703/what-is-the-cleanest-way-to-call-a-python-function-from-c-with-a-swig-wrapped