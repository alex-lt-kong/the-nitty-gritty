int sum_them_all(int* arr, int arr_len) {
    int sum = 0;
    for (int i = 0; i < arr_len; ++i) {
        sum = sum + arr[i];
    }
    return sum;
}