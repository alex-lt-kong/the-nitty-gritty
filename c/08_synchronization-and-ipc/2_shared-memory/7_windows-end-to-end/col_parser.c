#include <stdio.h>
#include <stdlib.h>

#include "common.h"

size_t read_ints(const char* file_name, int* arr_ptr, const size_t max_len) {
    int c;
    FILE* fptr;

    char line[1024]; // add line buffer and size tracking
    size_t col_count = 0;
    size_t line_count = 0; // initialize line_count

    if ((fptr = fopen(file_name, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", file_name);
        return 0;
    } 
    while ((c = getc(fptr)) != EOF){
        if (c == '\n') { // update nums on newline character
            line[col_count] = '\0'; // don't forget to terminate the string
            arr_ptr[line_count] = atoi(line); // atof() from stdlib.h is useful to convert string to number
            ++ line_count;
            if (line_count > max_len) {
                break;
            }
            col_count = 0; // start to read next line
        } else { // read line contents
            line[col_count] = (char)c;
            col_count = col_count + 1;
        }
    }
    fclose(fptr);
    return line_count; 
}

size_t read_dbls(const char* file_name, double* arr_ptr, const size_t max_len) {
    int c;
    FILE* fptr;

    char line[1024]; // add line buffer and size tracking
    size_t col_count = 0;
    size_t line_count = 0; // initialize line_count

    if ((fptr = fopen(file_name, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", file_name);
        return 0;
    } 
    while ((c = getc(fptr)) != EOF){
        if (c == '\n') { // update nums on newline character
            line[col_count] = '\0'; // don't forget to terminate the string
            arr_ptr[line_count] = atof(line); // atof() from stdlib.h is useful to convert string to number
            ++ line_count;
            if (line_count > max_len) {
                break;
            }
            col_count = 0; // start to read next line
        } else { // read line contents
            line[col_count] = (char)c;
            col_count = col_count + 1;
        }
    }
    fclose(fptr);
     
    return line_count; 
}

size_t read_chrs(const char* file_name, char* arr_ptr, const size_t max_len) {
    int c;
    FILE* fptr;

    char line[1024]; // add line buffer and size tracking
    size_t col_count = 0;
    size_t line_count = 0; // initialize line_count

    if ((fptr = fopen(file_name, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", file_name);
        return 0;
    } 
    while ((c = getc(fptr)) != EOF){
        if (c == '\n') { // update nums on newline character
            line[col_count] = '\0'; // don't forget to terminate the string
            strcpy(arr_ptr + (line_count * CHAR_COL_BUF_SIZE), line);
            ++ line_count;
            if (line_count > max_len) {
                break;
            }
            col_count = 0; // start to read next line
        } else { // read line contents
            if (col_count < CHAR_COL_BUF_SIZE - 1) {
                // -1 is for '\0'
                line[col_count] = (char)c;
                ++ col_count;
            }
        }
    }
    fclose(fptr);
    return line_count; 
}