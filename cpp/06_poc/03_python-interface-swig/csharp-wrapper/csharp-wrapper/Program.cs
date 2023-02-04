using System;
using System.ComponentModel;
using System.Diagnostics;

namespace MyProgram
{
    class NewIf : MyIf  // derived class (child)
    {
        public override void myfunc(Student stu)
        {            
            //Console.WriteLine("score1 is {0:0.00}", stu.score1);
        }
    }
    public class Program
    {
        

        public static void Main(string[] args)
        {
            NewIf myIf = new NewIf();
            UInt32 iter_count = 1000 * 1000 * 10;
            var start = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds();
            myIf.start(iter_count);
            var diff = new DateTimeOffset(DateTime.UtcNow).ToUnixTimeMilliseconds() - start;
            Console.WriteLine($"{diff / 1000.0} sec, {1.0 * iter_count / diff * 1000} calls / sec");
        }
    }

}


