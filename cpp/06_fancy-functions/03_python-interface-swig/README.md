# Python interface - SWIG

* This project is a continuation of its [C version](../../../c/04_fancy-functions/04_python-interface/3_calling-c-from-python-swig/)

* Get SWIG: `apt install swig`

* To generate SWIG wrapper code files from an interface file:
`swig -c++ -python mylibs.i`

## Windows

* Install `swig` and make sure that its `Lib` can be accessed by `swig.exe`.
    * Issue `swig.exe -swiglib` to check which path `swig.exe` tries to load
    `Lib`
* Generate SWIG wrapper code files.
* Run `python3 .\setup.py build`
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