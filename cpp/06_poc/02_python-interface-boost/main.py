import mylib
import time


class StudentHandlerA(object):
    def onStudentIterated(self, a):
        pass
    

class StudentHandlerB(object):
    count = 0
    def onStudentIterated(self, stu: mylib.Student):
        self.count += 1
        if self.count % 10000 == 0:
          assert isinstance(stu, mylib.Student)
          print(f'{stu.score1},{stu.score2},{stu.score3},{stu.score4}')
        pass

iter_count = 1_500_000

sha = mylib.StudentHandler(iter_count, StudentHandlerA())
sha.prepareStudentData()
start = time.time()
sha.start()
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')

shb = mylib.StudentHandler(iter_count, StudentHandlerB())
shb.prepareStudentData()
start = time.time()
shb.start()
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')

