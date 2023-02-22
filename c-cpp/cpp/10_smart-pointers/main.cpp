#include <memory>
#include <iostream>

using namespace std;

void dumb_pointer() {
    cout << "dumb_pointer() called\n";
    int* valuePtr = new int(15);
    int x = 45;

    if (x == 45)
        cout << "dumb_pointer() returned\n" << endl;
        return;   // here we have a memory leak, valuePtr is not deleted

    delete valuePtr;
}

void smart_pointer_unique_ptr()
{
    cout << "smart_pointer_unique_ptr() called\n";
    std::unique_ptr<int> valuePtr(new int(15));
    int x = 45;

    if (x == 45)
        cout << "smart_pointer_unique_ptr() returned\n" << endl;
        return;   // no memory leak anymore!

}

void smart_func_converted_from_dumb_func(size_t arr_size)
{
    cout << "smart_func_converted_from_dumb_func() called\n";
    struct FreeDeleter
    {
        void operator()(void *p) const
        {
            std::free(p);
        }
    };


    int* dynamic_int_arr = (int*)malloc(sizeof(int) * arr_size);
    // In reality, malloc() should be some deep-rooted C functions,
    // probably in a compiled so file. Here we simplify the scenario by
    // directly using a malloc()
    for (int i = 0; i < arr_size; i++) {
        dynamic_int_arr[i] = i;
    }
    std::unique_ptr<int[], FreeDeleter> smart_int_ptr(dynamic_int_arr);
    
    
    for (size_t i = 0; i < arr_size; ++i) {
        std::cout << smart_int_ptr[i] << std::endl;
    }
    cout << "smart_func_converted_from_dumb_func() returned\n" << endl;
}



int main() {
    dumb_pointer();
    smart_pointer_unique_ptr();
    smart_func_converted_from_dumb_func(12);
    return 0;
}
