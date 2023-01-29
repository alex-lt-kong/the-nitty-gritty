# Dependency injection and Autofac

* "Dependency injection" (or DI) sounds confusing but the idea is brutally
simple.

* Let's consider a naive C# sample with database operation:
    ```C#
    public class Example {
        private dataProvider myProvider;

        public Example() {
            myProvider = new dataProvider();
        }

        public void DoStuff() {
            // ...
            int[] data = myProvider.GetData();
            // ...
        }
    } 
    ```

* The first confusing point is that, what the heck does dependency mean? In
the above sample, does the code snippet show the `Example` class has any
"dependencies"?
    * The most common understandings of "dependency" in C# may include:
        1. The `Add Reference` -> `using MyLibrary;` stuff. Without these
        dependencies, a program can't even compile.
        2. Calling a RESTful endpoint, receiving messages from a message queue,
        etc. Without them, a program could run, but some functions may be broken.
    * But no, the "dependency" as in "dependency injection" means neither of
    them. It actually means the `myProvider` variable.
        * I think the argument goes this way: an object of the `Example` class
        needs an instance of `dataProvider` (i.e., `myProvider`) to work. So
        `myProvider` is also considered a "dependency".

* Okay, `myProvider` is the dependency. But why do we want to "inject" them?
    * Examining the example, one can notice that `myProvider.GetData()` is
    defined somewhere else. As long as `myProvider.GetData()` exists and it 
    returns an `int[]`, the program should work.
    * So theoretically different users (i.e., developers) can design their own
    `myProvider.GetData()`. For example, one user's `GetData()` may get data
    from a SQLite database, another user's `GetData()` may get data from an 
    RPC call.

* This requirement can be easily met by applying a naive OOP paradigm, with the
help of abstract class in C#:

    ```C#
    namespace MyProgram
    {
        public abstract class dataProvider
        {
            public abstract int[] GetData();
        }

        public class dataProviderSQLite : dataProvider
        {
            public override int[] GetData() {
                int[] data ={3, 1, 4, 1, 5, 9} ;
                return data;
            }
        }
        
        public class dataProviderREST : dataProvider
        {
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
            }
        }

    }
    ```
    ```
    Data from MyProgram.dataProviderSQLite@MyProgram.ExampleWithObject:
    3, 1, 4, 1, 5, 9, 
    Data from MyProgram.dataProviderREST@MyProgram.ExampleWithObject:
    6, 5, 5, 3, 6, 
    ```

    * With a bit more effort, we can reduce the lines of code needed from users
    by using reflection:

    ```C#
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
            public override int[] GetData() {
                int[] data ={3, 1, 4, 1, 5, 9} ;
                return data;
            }
        }
        
        public class dataProviderREST : dataProvider
        {
            public override int[] GetData() {
                int[] data = {6, 5, 5, 3, 6} ;
                return data;
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
                ExampleWithClassName example3 = new ExampleWithClassName("MyProgram.dataProviderSQLite");
                example3.DoStuff();
                ExampleWithClassName example4 = new ExampleWithClassName("MyProgram.dataProviderREST");
                example4.DoStuff();
            }
        }

    }
    ```
    ```
    Data from MyProgram.dataProviderSQLite@MyProgram.ExampleWithClassName:
    3, 1, 4, 1, 5, 9,
    Data from MyProgram.dataProviderREST@MyProgram.ExampleWithClassName:
    6, 5, 5, 3, 6, 
    ```

## References

* [James Shore - Dependency Injection Demystified](https://www.jamesshore.com/v2/blog/2006/dependency-injection-demystified)