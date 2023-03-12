#include <string>
#include <iostream>
#include <random>

#include "include/Currency.h"
#include "include/Market.h"
#include "include/MessageHeader.h"
#include "include/Quote.h"
#include "include/TradeData.h"

#include "../common.hpp"

using namespace std;
using namespace sbe;


const int messageHeaderVersion = 0;


size_t encodeTradeData(TradeDataStruct& tds, TradeData &td, char *buffer,
    uint64_t offset, uint64_t bufferLength) {
    td.wrapForEncode(buffer, offset, bufferLength)
        .quantity(tds.quantity)
        .putExchange(tds.exchange)
        .putSymbol(tds.symbol)
        .price(tds.price);

    if (strcmp(tds.currency, "USD") == 0) {
        td.currency(Currency::USD);
    } else if (strcmp(tds.currency, "EUR") == 0) {
        td.currency(Currency::EUR);
    } else {
        td.currency(Currency::HKD);
    }
    return td.encodedLength();
}

size_t decodeHdr(MessageHeader &hdr, char *buffer, uint64_t offset,
    uint64_t bufferLength) {
    hdr.wrap(buffer, offset, messageHeaderVersion, bufferLength);

    // decode the header
    /*
    cout << "messageHeader.blockLength=" << hdr.blockLength() << endl;
    cout << "messageHeader.templateId=" << hdr.templateId() << endl;
    cout << "messageHeader.schemaId=" << hdr.schemaId() << endl;
    cout << "messageHeader.schemaVersion=" << hdr.version() << endl;
    cout << "messageHeader.encodedLength=" << hdr.encodedLength() << endl;*/

    return hdr.encodedLength();
}

size_t decodeTradeData(
    TradeData &td,
    char *buffer,
    uint64_t offset,
    uint64_t actingBlockLength,
    uint64_t actingVersion,
    uint64_t bufferLength)
{
    td.wrapForDecode(buffer, offset, actingBlockLength, actingVersion, bufferLength);
    return td.encodedLength();
}


std::size_t encodeHdr(
    MessageHeader &hdr,
    TradeData &md,
    char *buffer,
    std::uint64_t offset,
    std::uint64_t bufferLength)
{
    // encode the header
    hdr.wrap(buffer, offset, messageHeaderVersion, bufferLength)
        .blockLength(TradeData::sbeBlockLength())
        .templateId(TradeData::sbeTemplateId())
        .schemaId(TradeData::sbeSchemaId())
        .version(TradeData::sbeSchemaVersion());

    return hdr.encodedLength();
}

int main() {
    srand(time(0));
    char buffer[BUFSIZE];
    MessageHeader hdr;
    TradeData td;
    size_t sampleIdx = rand() % ITER_COUNT;
    TradeDataStruct sampleTds;
    clock_t start, diff;
    start = clock();
    for (size_t i = 0; i < ITER_COUNT; ++i) {
        size_t encodeHdrLength = encodeHdr(hdr, td, buffer, 0, sizeof(buffer));
        size_t encodeMsgLength = encodeTradeData(
            tdsArr[i%(sizeof(tdsArr)/sizeof(struct TradeDataStruct))], td,
            buffer, hdr.encodedLength(), sizeof(buffer));

        MessageHeader hdrNew;
        TradeData tdNew;
        size_t decodeHdrLength = decodeHdr(hdrNew, buffer, 0, sizeof(buffer));
        size_t decodeMsgLength = decodeTradeData(tdNew, buffer,
            hdr.encodedLength(), hdr.blockLength(), hdr.version(), sizeof(buffer));
        if (sampleIdx == i) {
            memcpy(sampleTds.symbol, tdNew.symbol(), strlen(tdNew.symbol()));
            sampleTds.quantity = tdNew.quantity();
            sampleTds.price = tdNew.price();
            memcpy(sampleTds.exchange, tdNew.exchange(), strlen(tdNew.exchange()));
            if (tdNew.currency() == Currency::Value::USD) {
                memcpy(sampleTds.currency, "USD", 3);
            } else if (tdNew.currency() == Currency::Value::EUR) {
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