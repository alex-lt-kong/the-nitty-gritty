#include <memory>
#include <iostream>

using namespace std;

void standard_conversion_copy_only() {
    printf("Standard conversion - copy only\n");
    int16_t a = 65; //0x41
    int32_t b = a;
    uint8_t* i16_ptr = (uint8_t*)&a;
    uint8_t* i32_ptr = (uint8_t*)&b;
    printf("a: ");
    printf("%x\n", i16_ptr[0]);
    printf("b: ");
    printf("%02x", i32_ptr[0]);
    printf("%02x", i32_ptr[1]);
    printf("%02x", i32_ptr[2]);
    printf("%02x\n", i32_ptr[3]);
    // As we mostly use little-endian architecture system, the least
    // significant byte would be printf()'ed first
    printf("\n");
}

void standard_conversion_copy_plus() {
    printf("Standard conversion - copy plus bytes manipulation\n");
    float a = 3.14159;
    double b = a;
    double c = 3.14159;
    uint8_t* f32_ptr = (uint8_t*)&a;
    uint8_t* f64_ptr = (uint8_t*)&b;
    printf("a: ");
    for (size_t i = 0; i < sizeof(float); ++i) {
        printf("%02x", f32_ptr[i]);
    }
    printf("\nb: ");
    for (size_t i = 0; i < sizeof(double); ++i) {
        printf("%02x", f64_ptr[i]);
    }
    printf("\nc: ");
    f64_ptr = (uint8_t*)&c;
    for (size_t i = 0; i < sizeof(double); ++i) {
        printf("%02x", f64_ptr[i]);
    }
    printf("\n");
    printf("b == c: %d\n\n", b == c);
}

void standard_conversion_value_changed() {
    printf("Standard conversion - value changed\n");
    int32_t a = -1234567;
    uint32_t b = a;
    uint8_t* i32_ptr = (uint8_t*)&a;
    printf("a:   %d (0x", a);
    for (size_t i = 0; i < sizeof(int32_t); ++i) {
        printf("%02x", i32_ptr[i]);
    }
    printf(")\nb: %u (0x", b);
    i32_ptr = (uint8_t*)&b;    
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
        printf("%02x", i32_ptr[i]);
    }
    printf(")\n");
    
    uint64_t c = a;
    uint8_t* i64_ptr = (uint8_t*)&c;
    printf("c: %lu (0x", a);
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        printf("%02x", i64_ptr[i]);
    }
    printf(")\n\n");
}

int main() {
    standard_conversion_copy_only();
    standard_conversion_copy_plus();
    standard_conversion_value_changed();
    return 0;
}