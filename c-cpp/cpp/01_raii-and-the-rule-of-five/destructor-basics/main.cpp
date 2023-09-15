#include <cassert>
#include <cstring>
#include <iostream>

using namespace std;

class Wallet {
public:
  // In order to check the internal state of the class, we expose all member
  // variables to users
  int *ptr = nullptr;
  ssize_t wallet_size = -1;
  Wallet(ssize_t wallet_size, int *data) {
    if (wallet_size <= 0) {
      throw invalid_argument("wallet_size must be positive");
    }
    ptr = (int *)malloc(sizeof(int) * wallet_size);
    if (ptr == nullptr) {
      throw bad_alloc();
    }
    this->wallet_size = wallet_size;
    memcpy(ptr, data, wallet_size * sizeof(int));
  }

  int &operator[](size_t x) // C++23
  {
    if (x >= wallet_size) {
      throw out_of_range("");
    }
    return ptr[x];
  }

  ~Wallet() noexcept {
    cout << "Destructor called" << endl;
    free(ptr);
    ptr = nullptr;
    wallet_size = -1;
  }
};

int main(void) {
  int raw_data[] = {3, 1, 4, 1, 5, 9, 2, 6};
  Wallet myWallet = Wallet(sizeof(raw_data) / sizeof(int), raw_data);
  cout << myWallet.wallet_size << endl;
  for (int i = 0; i < myWallet.wallet_size; ++i) {
    cout << myWallet[i] << ",";
  }
  cout << endl;
  myWallet.~Wallet();
  assert(myWallet.wallet_size == -1);
  assert(myWallet.ptr == nullptr);
  cout << "main() about to return" << endl;
  return 0;
}