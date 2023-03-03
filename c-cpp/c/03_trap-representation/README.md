# Trap representation

* The definition of "trap representation" in [C11][1] is scant:
    > 3.19.4 trap representation
    > an object representation that need not represent a value of the object type

* Let's use `uint16_t` (commonly known as `unsigned short`) as an example to
demonstrate what's trap representation and how it helps:
    * The range of `uint16_t` is `[0, 65,535]`. The reason is that `uint16_t`
    has, well, 16 bits. If all 16 bits are `0`, we get 0. If all 16 bites are
    `1` we get 65535.
    * This works fine until we want to represent something like
    "uninitialized" (let's call this state `NaN` just for simplicity).
    `uint16_t` as we know can't be used to represent `NaN` as every
    possible bit combination is used to represent a number between 0 and 65535
    (all in big-endian just for convenience):
        * `0000 0000 0000 0000` represents 0
        * `0011 0000 0111 0000` represents 12400
        * `1011 1111 0111 1111` represents 49023
        * `1111 1111 1111 1111` represents 65535
        * etc.
    * One may think, how about we define a new two-byte long unsigned integer
    type called `my unsigned short` and shrink its range from `uint16_t`'s
    `[0, 65,535]` to `[0, 65,534]` so that one bits combination is spared,
    allowing us to use it to represent `NaN` like the following:
        * `0000 0000 0000 0000` represents 0
        * `1111 1111 1111 1110` represents 65534
        * `1111 1111 1111 1111` represents `NaN`
        * etc.
    * While it does seem reasonable for `1111 1111 1111 1111` to represent
    65535, we explicitly exclude it from the range of our new
    `my unsigned short` and make it represent something else.


* Since the C standard says unsigned integer types never overflow, for a
`my unsigned short` variable `a`, we should observe the following:

        ```C
        my unsigned short a = 65534;
        printf("%u\n", a);
        ++a;
        printf("%u\n", a);
        // 65534
        // 0
        ```
    * However, if we define a signed integer, `my short` in a
    similar manner (i.e., set its range to `[-32,768, 32,766]`), as signed
    integers can overflow, it may inadvertently get the variable `a` into
    the trap representation, causing UB:
        ```C
        my short a = 32766;
        ++a; // UB!
        ```

    * This is also one of the reasons (the other reason being the strict
    type aliasing rule) why we can't always cast a whatever type to integer
    (or whatever types) like this:
        ```C
        float* pi = 3.14;
        myint16_t* a = (myint16_t*)&pi;
        ```
        as it is not guaranteed that all possible bit combinations are used
        to represent a valid value

        * Well for unsigned integer types, not having any trap representation
        is mostly the case, but C standard doesn't guarantee this. It only
        guarantees that `unsigned char` must not have any trap representation.
    * Note that here we can only have `my unsigned short`, not `my_uint16_t` as
    section 7.20.1.1 of [C11][1] explicitly specifies that for `uint16_t`
    (or more generally, `uint*N*_t`) it has to be 16 bits (or more
    generally, `N` bits) long with no padding bits.

* The above example shows how we can use a valid bit pattern as trap
representation. This isn't always the case though.

* For example, when defining `my unsigned short`, instead of using
exactly 16 bits, internally we may choose to use 24 bits. `my unsigned short`'s
last two bytes are a normal `uint16_t` and the first byte is "reserved" for
all possible kinds of trap representations. For the sake of simplicity,
we define that a `my unsigned short` is only valid if the first bit is 0:

    * `0000 0000 0000 0000 0000 0000` represents 0
    * `0000 0000 0011 0000 0111 0000` represents 12400
    * `0000 0000 1011 1111 0111 1111` represents 49023
    * `0000 0000 1111 1111 1111 1111` represents 65535 
    * `1000 0000 0000 0000 0000 0000` represents `NaN`
    * `1000 0000 0011 0000 0111 0000` represents `NaN`
    * `1000 0000 1011 1111 0111 1111` represents `NaN`
    * `1000 0000 1111 1111 1111 1111` represents `NaN`
    * etc.

* This sort of trap representation is one potential use of "padding bits" in
section 6.2.6.2 of [C11][1]:
    > For unsigned integer types other than **unsigned char**, the bits
    > of the object representation shall be divided into two groups:
    > value bits and padding bits (there need not be any of the latter).
    > If there are *N* value bits, each bit shall represent a different
    > power of 2 between 1 and 2<sup>N-1</sup>, so that objects of that
    > type shall be capable of  representing values from 0 to
    > 2<sup>N</sup> − 1 using a pure binary representation; this shall be
    > known as the value representation. The values of any padding bits
    > are unspecified. <sup>53)</sup>
    
    ---

    > 53. Some combinations of padding bits might generate trap
    > representations, for example, if one padding bit is a parity bit.
    > Regardless, no arithmetic operation on valid values can generate
    > a trap representation other than as part of an exceptional
    > condition such as an overflow, and this cannot occur
    > with unsigned types. All other combinations of padding bits are
    > alternative object representations of the value specified by the
    > value bits.

    * Footnote 53 also clarified the meaning of these as valid representations:
        * `0111 0000 0000 0000 0000 0000` represents 0
        * `0000 1111 0011 0000 0111 0000` represents 12400
        * `0001 1000 1011 1111 0111 1111` represents 49023
        * `0000 0010 1111 1111 1111 1111` represents 65535 
        
        as "all other combinations of padding bits are alternative object
        representations of the value specified by the value bits.".

* One more interesting question about this padding bit/trap representation is:
for our wasteful but perfectly standard-conforming `my unsigned short`, what
would be the expected result of `sizeof`?:
    ```C
    my unsigned short a = 1234;
    printf("%u", sizeof a); // should it print 2 or 3?
    ```
    * Level I answer: it has to be 3. If the wording of C11 is not clear enough,
    we may think of it this way: for all sorts of offset-based operations,
    including:
    
        ```C
        my unsigned short arr[5] = {1, 2, 3, 4, 5};
        printf("%u\n", arr[2]);
        ```
        and
        ```C
        my unsigned short arr1[5] = {1, 2, 3, 4, 5};
        my unsigned short arr2[5];
        memcpy(arr2, arr1, 5);
        ```
        to work, the `sizeof` `my unsigned short` has to be 3.
    * Level II answer: the exact result depends on two parameters, how many
    bits are in `my unsigned short` and how many bits one byte has.
        * Section 5.2.4.2.1 of [C11][1] defines that `CHAR_BIT` stores how
        many bits should be in a byte.
        * The "correct" answer is three only under the assumption that
        `CHAT_BIT == 8`.

* TODO: compare trap representation with `NumPy`'s `NaN.
    
## References

1. [Draft of ISO C11 Standard][1]

[1]: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf "Draft of ISO C11 Standard"