# In Python, everything is an object. For people from the C world
# (including me), this may sound odd in the first place.
# But this is what the so-called "object-oriented programming" really should be.
# Python is not a strongly-typed language--when we define a variable,
# we don't have to (and we actually can't) give it a data type.
# This is consistent with the fact that everything is just an object--what
# can stop us from assigning an object with a different
# address and making it another "data type"?
# (However, when you think about it, you may start questioning what exactly
# does "strongly-typed"--for example, is C strongly-typed?
# If we are can to define a pointer in C, we are free to cast the value which
# the pointer is pointing to. In this sense, C is not that strongly-typed either!)

# Question: If everything in Python is an object, if we pass an object to a function, does it mean that in essence,
# only the pointer pointing to that object is passed to the function? Does it mean that if I change an object inside
# a function and we don't return the changed object, the original object will also be changed?
# The answer is: it depends

# a is passed to add_one() and we print() a again, the value is the same.
def add_one(num: int):
    num = num + 1
    print(f"num={num}")


a = 0
print(f"a={a}")
add_one(a)
print(f"a={a}")

# arr is passed to append_one() and we print() arr again, the value is changed!


def append_one(arr):
    arr.append(666)


arr = [0, 76, 1]
print(f"arr={arr}")
append_one(arr)
print(f"arr={arr}")


# Python makes a distinction between mutable and immutable objects. Simply put, Objects of built-in types like
#  (int, float, bool, str, tuple) are immutable. Objects of built-in types like (list, set, dict) are mutable.
# Here we only focus on one specific property of this distinction;
# If an object is considered an immutable type, well I will not say it is really pass by value,
# but at least it behaves the same as the traditional pass by value. That is, whatever happens
# to the value inside a function does not have an impact on the value outside of the function.
# On the other hand, if an object is mutable, the behavior is the same as passing a pointer in C--
# whatever we do to the object changes the outside object in the same way;
# There are a few cases in which mutability is not obvious:
# * datetime: immutable;
# * string: immutable (well this may be considered obvious by some, but in C, string is just an array of characters, so...);
# * tuple: immutable as a container. But its elements can be mutable.
