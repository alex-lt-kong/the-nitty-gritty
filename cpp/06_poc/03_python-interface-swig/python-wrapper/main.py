import mylib


class StudentHandlerA(mylib.StudentHandler):
    def onStudentIterated(self, a):
        pass
    

class StudentHandlerB(mylib.StudentHandler):
    def onStudentIterated(self, stu: mylib.Student):
        self.studentCount += 1
        if self.studentCount % 10000 == 0:
            print(stu.score1)
        pass

sha = StudentHandlerA()
import time
iter_count = 1_000_000
start = time.time()
sha.start(iter_count)
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')

shb = StudentHandlerB()
start = time.time()
shb.start(iter_count)
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')

