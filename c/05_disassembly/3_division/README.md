# Division

* `divide_by_2.asm`: `gcc` is smart enough to convert `divide by 2` to shift the operand right by one bit.

* `divide_by_constant.asm`: `gcc` is smart enough to convert `divide by 13` to firstly multiply by 1,321,528,399 and
then right shift the result by 34 bits. Painstakingly writing `a / 13` as `a * 0.07692307692307693` is unlikely
able to make the code faster.
