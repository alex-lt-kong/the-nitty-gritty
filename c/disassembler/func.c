int sum(int a, int b) {
    int temp = 0;
    temp = a + b;
    return temp;
}

int sum_them_all(int* arr, int arr_len) {
    int sum = 0;
    for (int i = 0; i < arr_len; ++i) {
        sum += arr[i];
    }
    return sum;
}