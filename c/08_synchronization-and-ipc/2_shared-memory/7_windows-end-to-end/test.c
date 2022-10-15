#include "col_parser.c"

int main(int argc, char* argv[]) {
    int* int_arr_ptr = malloc(sizeof(int) * 16);
    double* dbl_arr_ptr = malloc(sizeof(double) * 16);
    char* chr_arr_ptr = malloc(sizeof(char) * 16 * CHAR_COL_BUF_SIZE);
    size_t int_count = read_ints(".\\data\\col1_int.txt", int_arr_ptr, 16);
    for (int i = 0; i < int_count; ++i) {
        printf("%d, ", int_arr_ptr[i]);
    }
    printf("\n");

    size_t dbl_count = read_dbls(".\\data\\col2_dbl.txt", dbl_arr_ptr, 16);
    for (int i = 0; i < dbl_count; ++i) {
        printf("%lf, ", dbl_arr_ptr[i]);
    }
    printf("\n");
    
    size_t chr_count = read_chrs(".\\data\\col3_chr.txt", chr_arr_ptr, 16);
    for (int i = 0; i < chr_count; ++i) {
        printf("%s, ", chr_arr_ptr + i * CHAR_COL_BUF_SIZE);
    }
    printf("\n");

    free(int_arr_ptr);
    free(dbl_arr_ptr);
    free(chr_arr_ptr);
}