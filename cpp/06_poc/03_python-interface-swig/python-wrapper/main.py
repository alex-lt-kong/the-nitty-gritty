import mylib
import time


class DepartmentA(mylib.Department):
    def __init__(self):
        mylib.Department.__init__(self)
    def onStudentIterated(self, stu: mylib.Student):
        pass
    

class DepartmentB(mylib.Department):
    count: int
    def __init__(self):
        self.count = 0
        mylib.Department.__init__(self)
    
    def onStudentIterated(self, stu: mylib.Student):
        self.count += 1
        if self.count % 10000 == 0:
            assert isinstance(stu, mylib.Student)
            print(f'{stu.score1},{stu.score2},{stu.score3},{stu.score4}')


iter_count = 1_000_000

dept_b = DepartmentB()
dh = mylib.DepartmentHandler(dept_b, iter_count)
dh.prepareStudentData()
start = time.time()
dh.start()
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')

dept_a = DepartmentA()
dh = mylib.DepartmentHandler(dept_a, iter_count)
dh.prepareStudentData()
start = time.time()
dh.start()
diff = time.time() - start
print(f'{diff} sec ({iter_count / diff:,.0f} times / sec)')


