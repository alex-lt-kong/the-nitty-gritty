#include <random>

#include "string-demo.h"

using namespace std;


int main() {
    srand(time(NULL));

    ssoNaiveDemo();
    longStringNoSsoDemo();
    return 0;
}