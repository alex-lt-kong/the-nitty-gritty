using System;
using System.Diagnostics;

namespace MyProgram
{
    public abstract class dataProvider
    {
        public abstract int[] GetData();
    }

    public class dataProviderSQLite : dataProvider
    {
        // There isn't any SQLite backend of course, we name it this way...
        // just for fun.
        public override int[] GetData() {
            int[] data ={3, 1, 4, 1, 5, 9} ;
            return data;
        }
    }
    
    public class dataProviderREST : dataProvider
    {
        // There isn't any RESTful call of course, we name it this way...
        // just for fun.
        public override int[] GetData() {
            int[] data = {6, 5, 5, 3, 6} ;
            return data;
        }
    }
    
    public class ExampleWithObject {
        private dataProvider myProvider;

        public ExampleWithObject(dataProvider myProvider) {
            this.myProvider = myProvider;
        }

        public void DoStuff() {
            int[] data = myProvider.GetData();
            Console.WriteLine($"Data from {this.myProvider}@{this}:");
            for (int i = 0; i < data.Length; ++i) {
                Console.Write($"{data[i]}, ");
            }
            Console.WriteLine();
        }
    } 
    public class ExampleWithClassName {
        private dataProvider? myProvider;

        public ExampleWithClassName(string className) {
            Type t = Type.GetType(className); 
            myProvider = (dataProvider?)Activator.CreateInstance(t);
        }

        public void DoStuff() {
            int[] data = myProvider.GetData();
            Console.WriteLine($"Data from {this.myProvider}@{this}:");
            for (int i = 0; i < data.Length; ++i) {
                Console.Write($"{data[i]}, ");
            }
            Console.WriteLine();
        }
    } 

    public class Program
    {
        public static void Main(string[] args)
        {
            dataProviderSQLite myProvider1 = new dataProviderSQLite();
            ExampleWithObject example1 = new ExampleWithObject(myProvider1);
            example1.DoStuff();
            dataProviderREST myProvider2 = new dataProviderREST();
            ExampleWithObject example2 = new ExampleWithObject(myProvider2);
            example2.DoStuff();
                        
            ExampleWithClassName example3 = new ExampleWithClassName("MyProgram.dataProviderSQLite");
            example3.DoStuff();
            ExampleWithClassName example4 = new ExampleWithClassName("MyProgram.dataProviderREST");
            example4.DoStuff();
        }
    }

}


