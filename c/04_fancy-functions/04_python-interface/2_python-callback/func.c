#include <Python.h>

/* the C standard library qsort function, just as an example! */
extern void qsort(void *, size_t, size_t, int (*)(const void *, const void *));

/* static data (sigh), as we have no callback data in this (nasty) case */
static PyObject *py_compare_func = NULL;

static int
stub_compare_func(const void *cva, const void *cvb)
{
    int retvalue = 0;
    const PyObject **a = (const PyObject**)cva;
    const PyObject **b = (const PyObject**)cvb;

    // Build up the argument list...
    PyObject *arglist = Py_BuildValue("(OO)", *a, *b);

    // ...for calling the Python compare function
    PyObject *result = PyEval_CallObject(py_compare_func, arglist);

    if (result && PySet_Check(result)) {
        retvalue = PyLong_AsLong(result);
    }

    Py_XDECREF(result);
    Py_DECREF(arglist);

    return retvalue;
}

static PyObject *pyqsort(PyObject *obj, PyObject *args)
{
    PyObject *pycompobj;
    PyObject *list;
    if (!PyArg_ParseTuple(args, "OO", &list, &pycompobj))
        return NULL;

    // Make sure second argument is a function
    if (!PyCallable_Check(pycompobj)) {
        PyErr_SetString(PyExc_TypeError, "Need a callable object!");
    } else {
        // Save the compare function. This obviously won't work for multithreaded
        // programs and is not even a reentrant, alas -- qsort's fault!
        py_compare_func = pycompobj;
        if (PyList_Check(list)) {
            int size = PyList_Size(list);
            int i;

            // Make an array of (PyObject *), because qsort does not know about
            // the PyList object
            PyObject **v = (PyObject **) malloc( sizeof(PyObject *) * size );
            for (i=0; i<size; ++i) {
                v[i] = PyList_GetItem(list, i);
                // Increment the reference count, because setting the list 
                // items below will decrement the reference count
                Py_INCREF(v[i]);
            }
            qsort(v, size, sizeof(PyObject*), stub_compare_func);
            for (i=0; i<size; ++i) {
                PyList_SetItem(list, i, v[i]);
                // need not do Py_DECREF - see above
            }
            free(v);
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef qsortMethods[] = {
    { "qsort", pyqsort, METH_VARARGS },
    { NULL, NULL }
};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "qsort", "Some documentation",
    -1,
    qsortMethods
};

PyMODINIT_FUNC
PyInit_cos_module(void)
{
    return PyModule_Create(&cModPyDem);
}

#else

/* module initialization */
/* Python version 2 */

void initqsort(void) {
    PyObject *m;
    m = PyModule_Create("qsort", qsortMethods);
}


#endif
