#include <Python.h>

// Function 1: A simple 'hello world' function
static PyObject* helloworld(PyObject* self, PyObject* args)
{
    printf("Hello World\n");
    return Py_None;
}

static PyObject* call_arbitrary_pyfunc(PyObject* self, PyObject *args) {
    PyObject* func;
    int iter_count;
    int ok = PyArg_ParseTuple(args, "Oi", &func, &iter_count);
    for (long i = 0; i < iter_count; ++i)
        PyObject_CallOneArg(func, PyLong_FromLong(i));
    return Py_None;
    // plus whatever other code you need (e.g. reference counting, return value handling)
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "helloworld", helloworld, METH_NOARGS, "Prints Hello World" },
    { "call_arbitrary_pyfunc", call_arbitrary_pyfunc, METH_VARARGS, "Make a callback" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "mylib",
    "Test Module",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_mylib(void)
{
    return PyModule_Create(&myModule);
}
