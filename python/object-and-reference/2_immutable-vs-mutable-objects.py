# in Python, the following is also true:
# * Constant has it address too.
# * Integer variable is, just like other variables, just a reference to an object. In fact, in Python, all variables are references to objects!
# * The above is also closely related to the fact that most python objects (booleans, integers, floats, strings, and tuples) are immutable--after you create the object and assign some value to it, you can't modify that value.
# * Assignements to variable names aren't actually changing the objects themselves, but setting the reference to a new object.


print('All three should be different:')
print(id(103)) 
print(id(104))
print(id(105))
print()
a = 103
b = 103

print('These two should be the same:')
print(id(a))
print(id(b))
print()

print('These two should be different and the 2nd one should be the same as the above one:')
a += 1
print(id(a))
a -= 1
print(id(a))
a += 1