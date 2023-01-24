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
        private static IteratorWithInterface iteratorWithInterface = new IteratorWithInterface();
        private static IteratorWithoutInterface iteratorWithoutInterface = new IteratorWithoutInterface();
        private static IteratorWithInheritance iteratorWithInheritance = new IteratorWithInheritance();
        public void Iterate(UInt32 iterations)
        {
            for (int i = 0; i < iterations; ++i) {
                ++sum;
            }
        }
        [Benchmark]
        public void IterateWithInterface()
        {            
            iteratorWithInterface.Iterate(1024);
        }
        [Benchmark]
        public void IterateWithInheritance()
        {            
            iteratorWithInheritance.Iterate(1024);
        }

        [Benchmark]
        public void IterateWithoutInterface()
        {            
            iteratorWithoutInterface.Iterate(1024);
        }
        [Benchmark]
        public void IterateWithoutAnything()
        {            
            Iterate(1024);
        }
    }

    public class Program
    {    
        static UInt64 sum = 0;
        public static void Iterate(UInt32 iterations)
        {
            for (int i = 0; i < iterations; ++i) {
                ++sum;
            }
        }
       	public static long GetNanoseconds()
        {
            double timestamp = Stopwatch.GetTimestamp();
            double nanoseconds = 1_000_000_000.0 * timestamp / Stopwatch.Frequency;

            return (long)nanoseconds;
        }
        public static void Main(string[] args)
        {
            
            UInt32 TEST_COUNT = 10_000_000;
            IteratorWithInterface iteratorWithInterface = new IteratorWithInterface();
            IteratorWithoutInterface iteratorWithoutInterface = new IteratorWithoutInterface();
            IteratorWithInheritance iteratorWithInheritance = new IteratorWithInheritance();

            long start = GetNanoseconds();
            for (int i = 0; i < TEST_COUNT; ++i) {
                iteratorWithInterface.Iterate(1024);
            }
            long end = GetNanoseconds();
            Console.WriteLine($"IterateWithInterface: {(end - start) / TEST_COUNT} ns");

            start = GetNanoseconds();
            for (int i = 0; i < TEST_COUNT; ++i) {
                iteratorWithInheritance.Iterate(1024);
            }
            end = GetNanoseconds();
            Console.WriteLine($"IterateWithInheritance: {(end - start) / TEST_COUNT} ns");

            start = GetNanoseconds();
            for (int i = 0; i < TEST_COUNT; ++i) {
                iteratorWithoutInterface.Iterate(1024);
            }
            end = GetNanoseconds();
            Console.WriteLine($"IterateWithoutInterface: {(end - start) / TEST_COUNT} ns");

            start = GetNanoseconds();
            for (int i = 0; i < TEST_COUNT; ++i) {
                Iterate(1024);
            }
            end = GetNanoseconds();
            Console.WriteLine($"IterateWithoutAnything: {(end - start) / TEST_COUNT} ns");
            Console.ReadLine();
            BenchmarkRunner.Run<AbstractionCostTest>();
        }
    }

}


