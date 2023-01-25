using System;
using System.Diagnostics;

namespace MyProgram
{

    public class Program
    {
        public static int Factorial(int n) {
            int result = 1;
            while (n > 1) {
                result *= n--;
            }
            return result;
        }

        public static int GetConstantId() {
            int id = 1 + 2 + 3 + 4 + 5;
            return id;
        }
        public static void Main(string[] args)
        {
            int id = GetConstantId();
            int f4 = Factorial(4);
            Debugger.Break();
            Console.WriteLine($"GetId(): {id}");
            Console.WriteLine($"Factorial(): {f4}");
        }
    }

}


