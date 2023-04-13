#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int get_sum(int num, int depth, int max_depth) {
    printf("?\n");
	if (depth < max_depth) {
		return get_sum(num + depth, depth+1, max_depth);
	} else {
		return num + depth;
	}
}

int main(void) {
    int len, idx;
    scanf("%d %d", &len, &idx);
    printf("len=%d, idx=%d\n", len, idx);
    int* arr = (int*)malloc(len * sizeof(int));
    if (arr == NULL) {
        perror("malloc()");
        return 1;
    }
    
    if (idx < 0 || idx >= len) {
        fprintf(stderr, "idx out of bounds\n");
        return 2;
    }
    arr[idx] = idx;
    arr[idx+1] = len;
    printf("sum=%d\n", arr[idx] + arr[idx+1]);
    printf("quotient=%d\n", arr[idx] / arr[idx+1]);
    if (arr[idx+1] > 100) {
        fprintf(stderr, "potential stackoverflow\n");
        return 3;
    }
	printf("recursive sum=%d\n", get_sum(arr[idx], 0, arr[idx+1]));
    free(arr);
    return 0;
}
