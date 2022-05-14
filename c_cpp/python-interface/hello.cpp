#include <boost/python.hpp>
#include <iostream>


using namespace std;

char const* say_hi()
{
    return "Hi!";
}

void sum_a_list1(int limit) {

    long long int sum = 0;
    int* arr = (int*)calloc(limit, sizeof(int));
    for(int i=0; i<limit;i++){
        arr[i] = i;
    }
    for(int i=0; i<limit;i++){
        sum += arr[i];
    }
    cout << sum << endl;
    delete[] arr;
}

void sum_a_list2(boost::python::list list) {

    boost::python::ssize_t len = boost::python::len(list);
    long long int sum = 0;
    for(int i=0; i<len;i++){
        sum += boost::python::extract<int>(list[i]);        
    }
    cout << sum << endl;
}

BOOST_PYTHON_MODULE(hello)
{
    boost::python::def("say_hi", say_hi);
    boost::python::def("sum_a_list", sum_a_list1);    
}
