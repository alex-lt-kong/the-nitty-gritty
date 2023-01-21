#include <Python.h>

/* the C standard library qsort function, just as an example! */
extern void qsort (void *, size_t, size_t, int (*)(const void *, const void *));


static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "Some Name", "Some documentation",
    -1,
    NULL
};

PyMODINIT_FUNC
PyInit_cos_module(void)
{
    return PyModule_Create(&cModPyDem);
}

