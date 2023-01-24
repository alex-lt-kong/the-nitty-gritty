using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System;
using System.Diagnostics;

namespace MyBenchmarks
{
    public interface IIterator
    {
        void Iterate(UInt32 iterations);
    }

    public class VirtualIterator
    {
        public UInt64 sum = 0;
        public virtual void Iterate(UInt32 iterations)
        {
            Console.WriteLine("It doesnt matter what I do here, it will be overridden anyway");
            for (int i = 0; i < iterations; ++i) {
                ++sum;
            }
        }
    }

    public class IteratorWithInheritance: VirtualIterator
    {
        public void Iterate(UInt32 iterations)
        {
            for (int i = 0; i < iterations; ++i) {
                ++sum;
            }
        }
    }

    public class IteratorWithInterface : IIterator
    {
        public UInt64 sum = 0;
        public void Iterate(UInt32 iterations)
        {
            for (int i = 0; i < iterations; ++i) {
                ++sum;
            }
        }
    }

    public class IteratorWithoutInterface
    {
        public UInt64 sum = 0;
        public void Iterate(UInt32 iterations)
        {
            for (int i = 0; i < iterations; ++i) {
                ++sum;
            }
        }
    }
    
    public class AbstractionCostTest
    {
        public UInt64 sum = 0;
        private static IteratorWithInterface staticIteratorWithInterface = new IteratorWithInterface();
        private static IteratorWithoutInterface staticIteratorWithoutInterface = new IteratorWithoutInterface();
        private static IteratorWithInheritance staticIteratorWithInheritance = new IteratorWithInheritance();
        public void Iterate(UInt32 iterations)
        {
            for (int i = 0; i < iterations; ++i) {
                ++sum;
            }
        }

        [Benchmark]
        public void IterateWithInterface()
        {          
            IteratorWithInterface iteratorWithInterface = new IteratorWithInterface();
            iteratorWithInterface.Iterate(1024);
        }
        [Benchmark]
        public void IterateWithInheritance()
        {            
            IteratorWithInheritance iteratorWithInheritance = new IteratorWithInheritance();
            iteratorWithInheritance.Iterate(1024);
        }

        [Benchmark]
        public void IterateWithoutInterface()
        {
            IteratorWithoutInterface iteratorWithoutInterface = new IteratorWithoutInterface();
            iteratorWithoutInterface.Iterate(1024);
        }
        [Benchmark]
        public void IterateWithoutAnything()
        {            
            Iterate(1024);
        }
        [Benchmark]
        public void StaticIterateWithInterface()
        {            
            AbstractionCostTest.staticIteratorWithInterface.Iterate(1024);
        }
        [Benchmark]
        public void StaticIterateWithInheritance()
        {            
            AbstractionCostTest.staticIteratorWithInheritance.Iterate(1024);
        }

        [Benchmark]
        public void StaticIterateWithoutInterface()
        {            
            AbstractionCostTest.staticIteratorWithoutInterface.Iterate(1024);
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            BenchmarkRunner.Run<AbstractionCostTest>();
        }
    }

}


