#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include <iostream>

#define ARR_SIZE 65536

using namespace std;

class StaticWallet {
private:
    void mallocPtrs() {
        curr_ptr0 = (int*)malloc(sizeof(int) * ARR_SIZE);
        if (curr_ptr0 == NULL) {
            std::bad_alloc exception;
            throw exception;
        }
        curr_ptr1 = (int*)malloc(sizeof(int) * ARR_SIZE);
        if (curr_ptr1== NULL) {
            free(curr_ptr0);
            // Need to handle the already malloc()'ed pointer manually.
            std::bad_alloc exception;
            throw exception;
        }
    }
public:
    int* curr_ptr0;
    int* curr_ptr1;

    StaticWallet () {
        cout << "StaticWallet () called" << endl;
        mallocPtrs();
    }
    // Copy constructor
    StaticWallet(const StaticWallet& rhs) {
        mallocPtrs();  
        memcpy(curr_ptr0, rhs.curr_ptr0, sizeof(int) * ARR_SIZE);
        memcpy(curr_ptr1, rhs.curr_ptr1, sizeof(int) * ARR_SIZE);
        cout << "copy StaticWallet() called" << endl;
    }
    // Copy assignment operator
    StaticWallet& operator=(const StaticWallet& rhs) {
        memcpy(curr_ptr0, rhs.curr_ptr0, sizeof(int) * ARR_SIZE);
        memcpy(curr_ptr1, rhs.curr_ptr1, sizeof(int) * ARR_SIZE);
        cout << "operator=(const StaticWallet& rhs) called" << endl;
        return *this;
    }
    ~StaticWallet () {
        free(curr_ptr0);
        free(curr_ptr1);
    }
};

class DynamicWallet {
private:
    void mallocPtrs() {
        curr_ptr0 = (int*)malloc(sizeof(int) * wallet_size);
        if (curr_ptr0 == NULL) {
            std::bad_alloc exception;
            throw exception;
        }
        curr_ptr1 = (int*)malloc(sizeof(int) * wallet_size);
        if (curr_ptr1== NULL) {
            free(curr_ptr0);
            // Need to handle the already malloc()'ed pointer manually.
            std::bad_alloc exception;
            throw exception;
        }
    }
public:
    int* curr_ptr0;
    int* curr_ptr1;
    size_t  wallet_size;
    DynamicWallet (size_t wallet_size) {
        cout << "DynamicWallet () called" << endl;
        this->wallet_size = wallet_size;
        mallocPtrs();
    }
    // Copy constructor
    DynamicWallet(const DynamicWallet& rhs) {
        wallet_size = rhs.wallet_size;
        mallocPtrs();
        memcpy(curr_ptr0, rhs.curr_ptr0, sizeof(int) * wallet_size);
        memcpy(curr_ptr1, rhs.curr_ptr1, sizeof(int) * wallet_size);
        cout << "copy DynamicWallet() called" << endl;
    }
    // Copy assignment operator
    DynamicWallet& operator=(const DynamicWallet& rhs) {
        this->wallet_size = rhs.wallet_size;
        DynamicWallet t_wallet = DynamicWallet(rhs);
        free(curr_ptr0);
        free(curr_ptr1);
        mallocPtrs();
        memcpy(curr_ptr0, t_wallet.curr_ptr0, sizeof(int) * wallet_size);
        memcpy(curr_ptr1, t_wallet.curr_ptr1, sizeof(int) * wallet_size);
        cout << "operator=(const DynamicWallet& rhs) called" << endl;
        return *this;
    }
    ~DynamicWallet () {
        free(curr_ptr0);
        free(curr_ptr1);
    }
};

int main() {
    cout << "=== StaticWallet ===" << endl;
    StaticWallet first_wallet = StaticWallet();
    first_wallet.curr_ptr0[0] = 1;
    first_wallet.curr_ptr1[2] = 2147483647;
    cout << "first_wallet: " << first_wallet.curr_ptr0[0] << ", "
         << first_wallet.curr_ptr1[2] << endl << endl;

    StaticWallet second_wallet = first_wallet;
    second_wallet.curr_ptr0[0] = 31415926;
    second_wallet.curr_ptr1[2] = -1;
    cout << "first_wallet: " << first_wallet.curr_ptr0[0] << ", "
         << first_wallet.curr_ptr1[2] << endl;
    cout << "second_wallet: " << second_wallet.curr_ptr0[0] << ", "
         << second_wallet.curr_ptr1[2] << endl << endl;

    StaticWallet third_wallet = StaticWallet();
    third_wallet = first_wallet;
    third_wallet.curr_ptr0[0] = 666;
    third_wallet.curr_ptr1[2] = 2333;
    cout << "first_wallet: " << first_wallet.curr_ptr0[0] << ", "
         << first_wallet.curr_ptr1[2] << endl;
    cout << "second_wallet: " << second_wallet.curr_ptr0[0] << ", "
         << second_wallet.curr_ptr1[2] << endl;
    cout << "third_wallet: " << third_wallet.curr_ptr0[0] << ", "
         << third_wallet.curr_ptr1[2] << endl << endl << endl;


    cout << "=== DynamicWallet ===" << endl;
    DynamicWallet first_dwallet = DynamicWallet(2048);
    first_dwallet.curr_ptr0[0] = 3;
    first_dwallet.curr_ptr1[2047] = 666;
    cout << "first_dwallet: " << first_dwallet.curr_ptr0[0] << ", "
         << first_dwallet.curr_ptr1[2047] << endl << endl;

    DynamicWallet second_dwallet = first_dwallet;
    second_dwallet.curr_ptr0[0] = 31415926;
    second_dwallet.curr_ptr1[2047] = -1;
    cout << "first_dwallet: " << first_dwallet.curr_ptr0[0] << ", "
         << first_dwallet.curr_ptr1[2047] << endl;
    cout << "second_dwallet: " << second_dwallet.curr_ptr0[0] << ", "
         << second_dwallet.curr_ptr1[2047] << endl << endl;

    DynamicWallet third_dwallet = DynamicWallet(1);
    third_dwallet = first_dwallet;
    third_dwallet.curr_ptr0[0] = -123;

    cout << "first_dwallet: " << first_dwallet.curr_ptr0[0] << ", "
         << first_dwallet.curr_ptr1[2047] << endl;
    cout << "second_dwallet: " << second_dwallet.curr_ptr0[0] << ", "
         << second_dwallet.curr_ptr1[2047] << endl;
    cout << "third_dwallet: " << third_dwallet.curr_ptr0[0] << ", "
         << third_dwallet.curr_ptr1[2047] << endl << endl;


    first_dwallet = first_dwallet;
    cout << "first_dwallet: " << first_dwallet.curr_ptr0[0] << ", "
         << first_dwallet.curr_ptr1[2047] << endl;
    return 0;
}
