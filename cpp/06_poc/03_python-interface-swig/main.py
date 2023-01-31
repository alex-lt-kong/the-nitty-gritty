import mylibs

mc = mylibs.MyClass()
print(mc.Id)
print(mc.Name)
print(mc.PhoneNumber)
for i in range(5):
    print(mc.Scores[i], end=', ')
print('\n')

print('=== mc.Print()@mylib.so ===')
mc.Print()
