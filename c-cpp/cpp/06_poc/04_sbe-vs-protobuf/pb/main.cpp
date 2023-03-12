#include <iostream>
#include <string>
#include <iomanip>

#include "include/tradeData.pb.h"
#include "../common.hpp"

using namespace std;
using namespace pb;

void decodeBytesToStructProtoBuf(TradeData& td, string& byte_msg) {
    
    td.ParseFromString(byte_msg);
}

void encodeStructToBytesProtoBuf(TradeData& td, TradeDataStruct& tds,
  string* byte_msg) {
    td.Clear();
    td.set_symbol(tds.symbol);
    td.set_price(tds.price);
    td.set_quantity(tds.quantity);
    td.set_exchange(tds.exchange);
    if (strcmp(tds.currency, "USD") == 0) {
      td.set_currency(pb::TradeData::USD);
    } else if (strcmp(tds.currency, "EUR") == 0) {
      td.set_currency(pb::TradeData::EUR);
    } else {
      td.set_currency(pb::TradeData::HKD);
    }

    td.SerializeToString(byte_msg);
    // According to: https://stackoverflow.com/questions/4986673/c11-rvalues-and-move-semantics-confusion-return-statement/4986802#4986802
    // No special treatment is needed--compiler will apply move constructor
    // or other optimization techniques automatically.
}

int main() {    
    srand(time(0));
    string byte_msgs_protobuf;
    TradeData td;
    size_t sampleIdx = rand() % ITER_COUNT;
    TradeDataStruct sampleTds;
    clock_t start, diff;
    start = clock();
    for (size_t i = 0; i < ITER_COUNT; ++i) {
        encodeStructToBytesProtoBuf(td,
          tdsArr[i%(sizeof(tdsArr)/sizeof(struct TradeDataStruct))], &byte_msgs_protobuf);

        TradeData tdNew;
       decodeBytesToStructProtoBuf(tdNew, byte_msgs_protobuf);
        if (sampleIdx == i) {
            memcpy(sampleTds.symbol, tdNew.symbol().data(),
              strlen(tdNew.symbol().data()));
            sampleTds.quantity = tdNew.quantity();
            sampleTds.price = tdNew.price();
            memcpy(sampleTds.exchange, tdNew.exchange().data(),
              strlen(tdNew.exchange().data()));
            if (tdNew.currency() == TradeData::Currency::TradeData_Currency_USD) {
                memcpy(sampleTds.currency, "USD", 3);
            } else if (tdNew.currency() == TradeData::Currency::TradeData_Currency_EUR) {
                memcpy(sampleTds.currency, "EUR", 3);
            } else {
                memcpy(sampleTds.currency, "HKD", 3);
            }
        }
    }
    diff = clock() - start;
    printTradeDataStruct(sampleTds);
    printElaspsedTime(diff);
    return 0;
}