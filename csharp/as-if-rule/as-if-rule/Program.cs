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
        public static void Main(string[] args)
        {
            int f4 = Factorial(4);
            Debugger.Break();
        }
    }

}


