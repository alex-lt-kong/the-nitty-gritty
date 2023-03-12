#include <string>
#include <iostream>
#include <iomanip>

using namespace std;

#define BUFSIZE 4096
#define ITER_COUNT 10000000UL

struct TradeDataStruct {
    char symbol[6] = {0};
    float price;
    int32_t quantity;
    char exchange[7] = {0};
    char currency[4] = {0};
};

struct TradeDataStruct tdsArr[] = {
    {"MSFT", 3.1415, 666,   "NASDAQ", "USD"},
    {"0005", 1.414,  1,     "HKEX",   "HKD"},
    {"BP",   2.71,   65535, "LSE",    "EUR"},
    {"GS",   0.001,  -123,  "NYSE",   "USD"}
};

void printTradeDataStruct(const TradeDataStruct& tds) {
    cout << "===== Sample Output =====" << endl;
    cout << "Symbol: " << tds.symbol << "\n"
         << "Quantity: " << tds.quantity << "\n"
         << "Price: " << tds.price << "\n"
         << "Currency: " << tds.currency << "\n"
         << "Exchange: " << tds.exchange << "\n"
         << endl;
}

void printElaspsedTime(const clock_t diff) {
    cout << "Iterated " << ITER_COUNT / 1000 / 1000 << " mil times\n"
         << "Elapsed "<< diff / 1000.0 << "ms (" << ITER_COUNT * 1.0 / diff
         << " mil records per sec or "
         << setprecision(2) << 1000 * diff / ITER_COUNT
         << " ns per record)" << endl;
}