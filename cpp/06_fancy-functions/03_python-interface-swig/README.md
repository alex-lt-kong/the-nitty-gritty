# Python interface - SWIG

* This project is a continuation of its [C version](../../../c/04_fancy-functions/04_python-interface/3_calling-c-from-python-swig/)

## Linux

* Install SWIG: `apt install swig`
* To generate SWIG wrapper code files from an interface file:
`swig -c++ -python mylibs.i`

## Windows

* Install `swig` from [here](https://www.swig.org/download.html).
    * This step can cause confusion as the `swigwin` installer provided
     does not contain a `Lib` subdirectory. `swig.exe` will complain things
     like "Unable to find 'swig.swg'" if it does not find *.swg files in `Lib`.
    * We can download the `Lib` directory from SWIG's
    [official repository](https://github.com/swig/swig/tree/master/Lib)
    and set `SWIG_LIB` to `Lib`'s path to let `swig.exe` have the library files.
* Generate SWIG wrapper code files: `swig.exe -c++ -python mylibs.i`.
* Run `python3 .\setup.py build` to compile Python module.
* Play
```
cd build\lib.win-amd64-3.8
python3
>>> import mylibs
>>> mc = mylibs.MyClass()
>>> mc.Scores[2]
12.3
>>> mc.Print()
Id: 31415
Name: MyObjectName
PhoneNumber: 1234567890
>>>
```

## References

* https://www.swig.org/Doc3.0/SWIGDocumentation.html#Python_nn6
* https://stackoverflow.com/questions/8776328/swig-interfacing-c-library-to-python-creating-iterable-python-data-type-from/8828454#8828454
* https://stackoverflow.com/questions/12392703/what-is-the-cleanest-way-to-call-a-python-function-from-c-with-a-swig-wrapped