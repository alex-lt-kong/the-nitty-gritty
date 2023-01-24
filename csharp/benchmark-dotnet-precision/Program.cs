using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace MyBenchmarks
{
    
    public class Iterations
    {
        UInt64 sum = 0;

        [Benchmark]
        public void IterateManyPlusOne() {
            UInt64 sum = 0;
            for (int i = 0; i < 1_000_000_000; ++i) {
                for (int j = 0; j < 33; ++j) {
                    ++sum;
                }
            }
        }

        [Benchmark]
        public void IterateOnePlusOne() {
            UInt64 sum = 0;
            for (int i = 0; i < 33; ++i) {
                ++sum;
            }
        }

        [Benchmark]
        public void IterateMany() {
            UInt64 sum = 0;
            for (int i = 0; i < 1_000_000_000; ++i) {
                for (int j = 0; j < 32; ++j) {
                    ++sum;
                }
            }
        }

        [Benchmark]
        public void IterateOne() {
            UInt64 sum = 0;
            for (int i = 0; i < 32; ++i) {
                ++sum;
            }
        }

        public UInt64 GetSum() {
            return sum;
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            var summary0 = BenchmarkRunner.Run<Iterations>();
        }
    }

}


