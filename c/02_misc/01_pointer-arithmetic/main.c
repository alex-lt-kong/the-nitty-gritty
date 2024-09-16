#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

int main() {
    uint32_t arr[] = {0xDEADBEEF, 0xEEEEFFFF, 0x11111111};
    printf("%p:%x, %p:%x, %p:%x\n",
        (void*)arr,     *(arr),
        (void*)(arr+1), *(arr+1),
        (void*)(arr+2), *(arr+2));
    /* The below type-cast violates the strict aliasing rule! It is used to
    demonstrate the behavior of pointers only! */
    printf("%p:%08x, %p:%08x, %p:%08x, %p:%08x\n",
        (void*)((uint16_t*)arr)   , *( (uint16_t*)arr),
        (void*)((uint16_t*)arr+1), *(((uint16_t*)arr)+1),
        (void*)((uint16_t*)arr+2), *(((uint16_t*)arr)+2),
        (void*)((uint16_t*)arr+3), *(((uint16_t*)arr)+3));
    printf("%p:%08x, %p:%08x, %p:%08x, %p:%08x\n",
        (void*)((uint8_t*)arr),   *(((uint8_t*)arr)),
        (void*)((uint8_t*)arr+1), *(((uint8_t*)arr)+1),
        (void*)((uint8_t*)arr+2), *(((uint8_t*)arr)+3),
        (void*)((uint8_t*)arr+3), *(((uint8_t*)arr)+3));
    return 0;
}