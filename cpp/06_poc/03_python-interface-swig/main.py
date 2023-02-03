import mylibs

#class MySillyClass:
#    value = 0

#mc = mylibs.MyClass()
#print(mc.Id)
#print(mc.Name)
#print(mc.PhoneNumber)
#for i in range(5):
#    print(mc.Scores[i], end=', ')
#print('\n')

#print('=== mc.Print()@mylib.so ===')
#mc.Print()



class MyCl(mylibs.MyIf):
  def myfunc(self, a):
    #self.count += 1
    #print(f'myfunc@main.py: {a}/{self.count}')
    pass

cl = MyCl()
import time
iter_count = 1_000_000
start = time.time()
cl.start(iter_count)
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')

