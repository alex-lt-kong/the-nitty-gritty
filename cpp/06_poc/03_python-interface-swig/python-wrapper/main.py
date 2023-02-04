import mylib
import time


class StudentHandlerA(mylib.StudentHandler):
    def onStudentIterated(self, a):
        pass
    

class StudentHandlerB(mylib.StudentHandler):
    count = 0
    def onStudentIterated(self, stu: mylib.Student):
        self.count += 1
        if self.count % 10000 == 0:
            assert isinstance(stu, mylib.Student)
            print(f'{stu.score1},{stu.score2},{stu.score3},{stu.score4}')


iter_count = 1_000_000

sha = StudentHandlerA(iter_count)
sha.prepareStudentData()
start = time.time()
sha.start()
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')

shb = StudentHandlerB(iter_count)
shb.prepareStudentData()
start = time.time()
shb.start()
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')

