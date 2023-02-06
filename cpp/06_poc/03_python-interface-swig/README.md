# Python interface - SWIG

* This project is a continuation of its [C version](../../../c/04_poc/04_python-interface/3_calling-c-from-python-swig/)
but it is much more complicated.

* We would like to achieve so-called [cross language polymorphism](https://www.swig.org/Doc4.1/Python.html#Python_directors)
with SWIG.
    * In practice, this fancy term means that we can define a base class in
    C++, inherit it in target languages (e.g., Python/C#/etc), override its
    methods in target languages and call the overridden method in C++:

        * `myclass.cpp`

        ```C++
        class MyClass {
        public:
            virtual void onCallback() = 0; // A virtual method to be overridden
            void start() {
                for (int i = 0; i < 3; ++i) {
                    onCallback(); // To be overridden by target languages
                }
            }
        };
        ```

        * `myclass.py`

        ```Python
        class MyPhthonClass(MyClass): # inherit MyClass from C++ "natively"
            def onCallback(self): # Override virtual method
                print('Hello world from Python')
        
        mpc = MyPhthonClass()
        mpc.start() # defined in C++, call transparently.
        # >>> Hello world from Python
        # >>> Hello world from Python
        # >>> Hello world from Python
        ```

        * `myclass.cs`

        ```C#
        class MyCSharpClass : MyClass { // inherit MyClass from C++ "natively"    
            public override void onCallback() { // override virtual method
                Console.WriteLine("Hello world from CSharp");
            }
        }
        public static void Main(string[] args) {
            MyCSharpClass mcc = new MyCSharpClass();
            mcc.start(); // defined in C++, call transparently.
        }
        // >>> Hello world from CSharp
        // >>> Hello world from CSharp
        // >>> Hello world from CSharp
        ```

## Linux

* Install SWIG: `apt install swig`
* To generate SWIG wrapper code files from an interface file:
`swig -c++ -python mylibs.i`
* `make`
* `python3 ./python-wrapper/main.py`

## Windows (MinGW)

* Install `swig` from [here](https://www.swig.org/download.html).
    * This step can cause confusion as the `swigwin` installer
     does not contain a `Lib` subdirectory. `swig.exe` will complain things
     like "Unable to find 'swig.swg'" if `Lib` directory is not found.
    * We can download the `Lib` directory from SWIG's
    [official repository](https://github.com/swig/swig/tree/master/Lib)
    and set `SWIG_LIB` to `Lib`'s path to let `swig.exe` have the library files.
* `make windows`
* Open `csharp-wrapper` in Visual Studio and build/run.

## References

* [stackoverflow.com - SWIG interfacing C library to Python (Creating 'iterable' Python data type from C 'sequence' struct](https://stackoverflow.com/questions/8776328/swig-interfacing-c-library-to-python-creating-iterable-python-data-type-from/8828454#8828454)
* [stackoverflow.com - What Is The Cleanest Way to Call A Python Function From C++ with a SWIG Wrapped Object](https://stackoverflow.com/questions/12392703/what-is-the-cleanest-way-to-call-a-python-function-from-c-with-a-swig-wrapped)
* [stackoverflow.com - SWIG+c+Python: Passing and receiving c arrays](https://stackoverflow.com/questions/36222455/swigcpython-passing-and-receiving-c-arrays)
* [stackoverflow.com - How to wrap a c++ function which takes in a function pointer in python using SWIG](https://stackoverflow.com/questions/22923696/how-to-wrap-a-c-function-which-takes-in-a-function-pointer-in-python-using-swi)
* [stackoverflow.com - How can I implement a C++ class in Python, to be called by C++?](https://stackoverflow.com/questions/9040669/how-can-i-implement-a-c-class-in-python-to-be-called-by-c)
* [Example for SWIG to wrap C++ library in .Net 6](https://iamsorush.com/posts/cpp-csharp-swig/)
* [SWIG 4.0 - Python](https://www.swig.org/Doc4.0/Python.html)