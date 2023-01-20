#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>

#define ARR_SIZE 65536

using namespace std;

class Wallet {
private:
    void mallocPtrs() {
        this->curr_ptr0 = (int*)malloc(sizeof(int) * ARR_SIZE);
        if (this->curr_ptr0 == NULL) {
            std::bad_alloc exception;
            throw exception;
        }
        this->curr_ptr1 = (int*)malloc(sizeof(int) * ARR_SIZE);
        if (this->curr_ptr1== NULL) {
            free(this->curr_ptr0);
            // Need to handle the already malloc()'ed pointer manually.
            std::bad_alloc exception;
            throw exception;
        }
    }
public:
    int* curr_ptr0;
    int* curr_ptr1;


    Wallet () {
        printf("Wallet () called\n");
        mallocPtrs();
    }
    // Copy constructor
    Wallet(const Wallet& w1) {
        mallocPtrs();  
        memcpy(this->curr_ptr0, w1.curr_ptr0, sizeof(int) * ARR_SIZE);
        memcpy(this->curr_ptr1, w1.curr_ptr1, sizeof(int) * ARR_SIZE);
        printf("copy Wallet() called\n");
    }
    // Copy assignment operator
    Wallet& operator=(const Wallet& w1) {
        memcpy(this->curr_ptr0, w1.curr_ptr0, sizeof(int) * ARR_SIZE);
        memcpy(this->curr_ptr1, w1.curr_ptr1, sizeof(int) * ARR_SIZE);
        printf("operator=(const Wallet& w1) called\n");
        return *this;
    }
    ~Wallet () {
        free(this->curr_ptr0);
        free(this->curr_ptr1);
    }
};

int main() {
    Wallet first_wallet = Wallet();
    first_wallet.curr_ptr0[0] = 1;
    first_wallet.curr_ptr1[2] = 2147483647;
    printf("first_wallet: %d, %d\n",
        first_wallet.curr_ptr0[0], first_wallet.curr_ptr1[2]);
    Wallet second_wallet = first_wallet;
    second_wallet.curr_ptr0[0] = 31415926;
    second_wallet.curr_ptr1[2] = -1;
    printf("first_wallet: %d, %d\n",
        first_wallet.curr_ptr0[0], first_wallet.curr_ptr1[2]);
    printf("second_wallet: %d, %d\n",
        second_wallet.curr_ptr0[0], second_wallet.curr_ptr1[2]);
    Wallet third_wallet = Wallet();
    third_wallet = first_wallet;
    third_wallet.curr_ptr0[0] = 666;
    third_wallet.curr_ptr1[2] = 2333;
    printf("first_wallet: %d, %d\n",
        first_wallet.curr_ptr0[0], first_wallet.curr_ptr1[2]);
    printf("second_wallet: %d, %d\n",
        second_wallet.curr_ptr0[0], second_wallet.curr_ptr1[2]);
    printf("third_wallet: %d, %d\n",
        third_wallet.curr_ptr0[0], third_wallet.curr_ptr1[2]);
    return 0;
}
