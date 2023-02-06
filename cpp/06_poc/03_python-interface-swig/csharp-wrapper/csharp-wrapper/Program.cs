using System;
using System.ComponentModel;
using System.Diagnostics;


namespace MyProgram
{
    class DepartmentA : Department
    {
        public override void onStudentIterated(Student stu)
        {
        }
    }
    class DepartmentB : Department
    {
        UInt32 count = 0;
        public override void onStudentIterated(Student stu)
        {
            ++count;
            if (count % 10000 == 0)
            {
                Console.WriteLine($"{stu.score1},{stu.score2},{stu.score3},{stu.score4}"); 
            }
        }
    }
    public class Program
    {
        

        public static void Main(string[] args)
        {
            DepartmentA dept = new DepartmentA();
            UInt32 iter_count = 1000 * 1000;
            DepartmentHandler deptHdl = new DepartmentHandler(dept, iter_count);
            deptHdl.prepareStudentData();
            var start = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds();
            deptHdl.start();
            var diff = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds() - start;
            Console.WriteLine($"{diff / 1000.0} sec, {1.0 * iter_count / diff * 1000} calls / sec");
        }
    }

}


