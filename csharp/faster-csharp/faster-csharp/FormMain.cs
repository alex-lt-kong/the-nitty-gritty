using System.Collections;
using System.Diagnostics;
using System.Buffers;
using System.Collections.Generic;
using System.Collections;
using System.Text;
using Mytest;
using Grpc.Core;
using System.Net;
using System;
using System.Net.Http.Headers;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;

namespace faster_csharp
{
    public partial class FormMain : Form
    {
        public FormMain()
        {
            InitializeComponent();
        }

        private void buttonGCvsLoop_Click(object sender, EventArgs e)
        {
            this.textBoxOutput.Text += $"===== GC.Collect() vs Loop ====={Environment.NewLine}";
            var watch = System.Diagnostics.Stopwatch.StartNew();
            /* Beginning with C# 3, variables that are declared at method scope can have an
             * implicit "type" var. An implicitly typed local variable is strongly typed just
             * as if you had declared the type yourself, but the compiler determines the type. 
             * For this particular case, var watch is equal to System.Diagnostics.Stopwatch watch
             */
            for (int i = 0; i < 1000; i++)
            {
                long a = 0;
                for (int j = 0; j < 50 * 1000; j++)
                {
                    a += j;
                }
            }
            watch.Stop();
            this.textBoxOutput.Text += $"50,000 iterations: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();

            watch.Restart();
            for (int i = 0; i < 1000; i++)
            {
                System.GC.Collect();
            }
            this.textBoxOutput.Text += $"GC.Collect(): {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            this.textBoxOutput.Text += $"\"a GC.Collect() call is worth 50 thousand iterations\"{Environment.NewLine}";
        }

        private void buttonStaticVSDynamicArraies_Click(object sender, EventArgs e)
        {
            this.textBoxOutput.Text += $"===== Static vs Dynamic Arraies ====={Environment.NewLine}";

            const int NumberOfItems = 10 * 1000 * 1000;
            var watch = System.Diagnostics.Stopwatch.StartNew();
            int[] vanillaArray = new int[NumberOfItems];
            for (int i = 0; i < NumberOfItems; i++)
            {
                vanillaArray[i] = i;
            }
            watch.Stop();
            this.textBoxOutput.Text += $"vanillaArray: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();
            System.GC.Collect();

            watch.Restart();
            ArrayList preAllocatedArrayList = new ArrayList();
            for (int i = 0; i < NumberOfItems; i++)
            {
                preAllocatedArrayList.Add(i);
            }
            watch.Stop();
            this.textBoxOutput.Text += $"preAllocatedArrayList: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();
            System.GC.Collect();

            watch.Restart();
            ArrayList naiveArrayList = new ArrayList(NumberOfItems);
            for (int i = 0; i < NumberOfItems; i++)
            {
                naiveArrayList.Add(i);
            }
            this.textBoxOutput.Text += $"naiveArrayList: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";

        }

        private void buttonArrayVSArrayPool_Click(object sender, EventArgs e)
        {
            this.textBoxOutput.Text += $"===== Array vs ArrayPool ====={Environment.NewLine}";
            const int NumberOfItems = 10000;
            Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < 1000; i++)
            {
                int[] array = new int[NumberOfItems];
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Array: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();

            watch.Restart();
            // ArrayPool does not appear to be supported in .NET Framework 4.8
            for (int i = 0; i < 100000; i++)
            {
                var pool = ArrayPool<int>.Shared;
                int[] array = pool.Rent(NumberOfItems);
                // need to remember is that it has a default max array length, equal to 2^20 (1024*1024 = 1 048 576).
                try { }                
                finally { // so we make sure an array is always returned
                    pool.Return(array);
                }
                // Returns an array to the pool that was previously obtained using the Rent(Int32) method on the same ArrayPool<T> instance.
                /* Once a buffer has been returned to the pool, the caller gives up all
                 * ownership of the buffer and must not use it. The reference returned from
                 * a given call to the Rent method must only be returned using the Return method once.*/
            }
            this.textBoxOutput.Text += $"ArrayPool: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
        }

        private void buttonStructVSClass_Click(object sender, EventArgs e)
        {
            this.textBoxOutput.Text += $"===== Class vs Struct vs Dict ====={Environment.NewLine}";
            const int NumberOfItems = 1000 * 1000;
            Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
            MyClass[] myClasses = new MyClass[NumberOfItems];
            for (int i = 0; i < NumberOfItems; i++)
            {
                myClasses[i] = new MyClass();
                myClasses[i].A = i + 1;
                myClasses[i].B = i + 2;
                myClasses[i].C = i + 3;
                myClasses[i].X = i + 4;
                myClasses[i].Y = i + 5;
                myClasses[i].Z = i + 6;
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Class: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();

            watch.Restart();
            MyStruct[] myStructs = new MyStruct[NumberOfItems];
            for (int i = 0; i < NumberOfItems; i++)
            {
                myStructs[i] = new MyStruct();
                myStructs[i].A = i + 1;
                myStructs[i].B = i + 2;
                myStructs[i].C = i + 3;
                myStructs[i].X = i + 4;
                myStructs[i].Y = i + 5;
                myStructs[i].Z = i + 6;
            }
            this.textBoxOutput.Text += $"Struct: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();

            watch.Restart();
            Dictionary<string, int>[] myDicts = new Dictionary<string, int>[NumberOfItems];
            for (int i = 0; i < NumberOfItems; i++)
            {
                myDicts[i] = new Dictionary<string, int>();
                myDicts[i]["A"] = i + 1;
                myDicts[i]["B"] = i + 2;
                myDicts[i]["C"] = i + 3;
                myDicts[i]["X"] = i + 4;
                myDicts[i]["Y"] = i + 5;
                myDicts[i]["Z"] = i + 6;
            }
            this.textBoxOutput.Text += $"Dictionary: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
        }

        private void buttonTryVSNoTry_Click(object sender, EventArgs e)
        {
            this.textBoxOutput.Text += $"===== Try vs No try ====={Environment.NewLine}";

            const int NumberOfItems = 10 * 1000 * 1000;
            var watch = System.Diagnostics.Stopwatch.StartNew();

            for (int i = 0; i < NumberOfItems; i++)
            {
                try
                {
                    int a = i;
                    int b = a + 1;
                }
                catch { }
                finally { }
            }
            watch.Stop();
            this.textBoxOutput.Text += $"try{{}}ed: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();
            System.GC.Collect();

            watch.Restart();
            for (int i = 0; i < NumberOfItems; i++)
            {
                int a = i;
                int b = a + 1;
            }
            watch.Stop();
            this.textBoxOutput.Text += $"not try{{}}ed: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            this.textBoxOutput.Text += $"\"There IS harm in try{{}}ing\"{Environment.NewLine}";
        }

        private void buttonFinalizerVSNoFinalizer_Click(object sender, EventArgs e)
        {
            this.textBoxOutput.Text += $"===== Finalizer vs No Finalizer ====={Environment.NewLine}";
            const int NumberOfItems = 1000 * 1000;
            Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
            MyClass[] myClasses = new MyClass[NumberOfItems];
            for (int i = 0; i < NumberOfItems; i++)
            {
                myClasses[i] = new MyClass();
                myClasses[i].A = i + 1;
                myClasses[i].B = i + 2;
                myClasses[i].C = i + 3;
                myClasses[i].X = i + 4;
                myClasses[i].Y = i + 5;
                myClasses[i].Z = i + 6;
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Class: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();

            watch.Restart();
            MyClassWithFinalizer[] myClassWithFinalizers = new MyClassWithFinalizer[NumberOfItems];
            for (int i = 0; i < NumberOfItems; i++)
            {
                myClassWithFinalizers[i] = new MyClassWithFinalizer();
                myClassWithFinalizers[i].A = i + 1;
                myClassWithFinalizers[i].B = i + 2;
                myClassWithFinalizers[i].C = i + 3;
                myClassWithFinalizers[i].X = i + 4;
                myClassWithFinalizers[i].Y = i + 5;
                myClassWithFinalizers[i].Z = i + 6;
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Class with a finalizer: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
        }

        private void buttonStringVSStringBuilder_Click(object sender, EventArgs e)
        {
            /* Strings are immutable. So whenever you add two string objects, a new
             * string object is created that holds the content of both strings. You
             * can avoid the allocation of memory for this new string object by taking
             * advantage of StringBuilder. */

            /* StringBuilder will improve performance in cases where you make repeated
             * modifications to a string or concatenate many strings together. However,
             * you should keep in mind that regular concatenations are faster than StringBuilder
             * for a small number of concatenations. */
            this.textBoxOutput.Text += $"===== String vs StringBuilder ====={Environment.NewLine}";

            const int NumberOfItems =  100 * 1000;
            var watch = new Stopwatch();

            watch.Start();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < NumberOfItems; i++)
            {
                sb.Append(i.ToString());                
            }
            watch.Stop();
            this.textBoxOutput.Text += $"StringBuilder: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();
            System.GC.Collect();

            watch.Restart();
            String str = "";
            for (int i = 0; i < NumberOfItems; i++)
            {
                str += i.ToString();
            }
            watch.Stop();
            this.textBoxOutput.Text += $"String: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
        }

        private void buttonSOHvsLOH_Click(object sender, EventArgs e)
        {
            this.textBoxOutput.Text += $"===== Small Object Heap vs Large Object Heap ====={Environment.NewLine}";

            const int NumberOfItems = 85 * 1000 / 4;
            const int upper = 100 * 1000;

            GC.Collect();

            Stopwatch watch = new Stopwatch();
            
            watch.Start();            
            for (int i = 0; i < upper; i++) {
                int[] smallArray = new int[NumberOfItems - 8];
                Debug.Assert(GC.GetGeneration(smallArray) == 0);
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Small Object Heap: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();
            System.GC.Collect();

            watch.Restart();            
            for (int i = 0; i < upper; i++) {
                int[] bigArray = new int[NumberOfItems];
                Debug.Assert(GC.GetGeneration(bigArray) == 2);
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Big Object Heap: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
        }

        private void buttonHashtableVSDictionary_Click(object sender, EventArgs e)
        {  
            this.textBoxOutput.Text += $"===== Hashtable vs Dictionary ====={Environment.NewLine}";
            const int NumberOfItems = 10 * 1000;
            Stopwatch watch = Stopwatch.StartNew();

            Hashtable[] myHashtable = new Hashtable[NumberOfItems];
            for (int i = 0; i < NumberOfItems; i++)
            {
                myHashtable[i] = new Hashtable();
                myHashtable[i]["A"] = i + 1;
                myHashtable[i]["B"] = i + 2;
                myHashtable[i]["C"] = i + 3;
                myHashtable[i][1] = i + 4;
                myHashtable[i][2] = i + 5;
                myHashtable[i][3] = i + 6;
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Same-type hashtable: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();

            GC.Collect();

            watch.Restart();
            myHashtable = new Hashtable[NumberOfItems];
            for (int i = 0; i < NumberOfItems; i++)
            {
                myHashtable[i] = new Hashtable();
                myHashtable[i]["A"] = new int[i + 1];
                myHashtable[i]["B"] = new string[i + 2];
                myHashtable[i]["C"] = new object();
                myHashtable[i][1] = i + 4;
                myHashtable[i][2] = i + 5;
                myHashtable[i][3] = i + 6;
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Different-type hashtable: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();

            GC.Collect();

            watch.Restart();
            Dictionary<string, int>[] myDicts = new Dictionary<string, int>[NumberOfItems];
            for (int i = 0; i < NumberOfItems; i++)
            {
                myDicts[i] = new Dictionary<string, int>();
                myDicts[i]["A"] = i + 1;
                myDicts[i]["B"] = i + 2;
                myDicts[i]["C"] = i + 3;
                myDicts[i]["1"] = i + 4;
                myDicts[i]["2"] = i + 5;
                myDicts[i]["3"] = i + 6;
            }
            this.textBoxOutput.Text += $"Dictionary: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
        }

        private void buttonDivideVSMultiplyByReciprocal_Click(object sender, EventArgs e)
        {
            this.textBoxOutput.Text += $"===== Division vs Reciprocal Multiplication ====={Environment.NewLine}";
            int upper = 100 * 1000 * 1000;
            double pi = 3.0;
            double piReciprocal = 1 / pi;

            Stopwatch watch = Stopwatch.StartNew();
            for (int i = 0; i < upper; i++)
            {
                double a = i / pi;
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Naive division: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";

            Application.DoEvents();
            GC.Collect();

            watch.Restart();
            for (int i = 0; i < upper; i++)
            {
                double a = i * piReciprocal;
            }
            this.textBoxOutput.Text += $"Naive reciprocal multiplication: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";

            Application.DoEvents();
            GC.Collect();

            watch.Restart();
            for (int i = 0; i < upper; i++)
            {
                if (i % 13 == 0) { continue; } // So that the results won't be easily vectorized.
                double a = i / pi;
            }
            watch.Stop();
            this.textBoxOutput.Text += $"Obfuscated division: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";

            watch.Restart();
            for (int i = 0; i < upper; i++)
            {
                if (i % 13 == 0) { continue; }
                double a = i * piReciprocal;
            }
            this.textBoxOutput.Text += $"Obfuscated reciprocal multiplication: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
        }

        private void button1_Click(object sender, EventArgs e)
        {     
            int upper = 10;
            Random rnd = new Random(2021);
            this.textBoxOutput.Text += $"===== gRPC vs RESTful API ====={Environment.NewLine}";

            Stopwatch watch = Stopwatch.StartNew();
            Channel channel = new Channel("127.0.0.1:50051", ChannelCredentials.Insecure);
            var client = new Calculator.CalculatorClient(channel);
            for (int i = 0; i < upper; i++)
            {
                double num1 = rnd.Next(1000);
                double num2 = rnd.Next(1000);
                CalculatorRequest gRPCrequest = new CalculatorRequest { Num1 = num1, Num2 = num2 };
                var replyAdd = client.add(gRPCrequest);
                var replyMinus = client.minus(gRPCrequest);
                Console.WriteLine($"num1: {num1}, num2: {num2}, sum: {replyAdd.Result}, diff: {replyMinus.Result}");
            }
            this.textBoxOutput.Text += $"gRPC: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
            Application.DoEvents();
            GC.Collect();
            
            watch.Restart();
            HttpClient httpClient = new HttpClient();
            httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
            for (int i = 0; i < upper; i++)
            {
                double num1 = rnd.Next(1000);
                double num2 = rnd.Next(1000);

                string url = $"http://localhost:50052/add/?num1={num1}&num2={num2}";
                HttpResponseMessage response = httpClient.GetAsync(url).Result;
                var webRequest = new HttpRequestMessage(HttpMethod.Get, url);
                JObject jsonSum = (JObject)JsonConvert.DeserializeObject(response.Content.ReadAsStringAsync().Result);

                url = $"http://localhost:50052/minus/?num1={num1}&num2={num2}";
                response = httpClient.GetAsync(url).Result;
                webRequest = new HttpRequestMessage(HttpMethod.Get, url);
                JObject jsonDiff = (JObject)JsonConvert.DeserializeObject(response.Content.ReadAsStringAsync().Result);
                Console.WriteLine($"num1: {num1}, num2: {num2}, sum: {jsonSum["result"]}, diff: {jsonDiff["result"]}");

            }
            this.textBoxOutput.Text += $"RESTful API: {watch.ElapsedMilliseconds} ms{Environment.NewLine}";
        }
    }

    class MyClass
    {
        public int A { get; set; }
        public int B { get; set; }
        public int C { get; set; }
        public int X { get; set; }
        public int Y { get; set; }
        public int Z { get; set; }
    }

    class MyClassWithFinalizer
    {
        public int A { get; set; }
        public int B { get; set; }
        public int C { get; set; }
        public int X { get; set; }
        public int Y { get; set; }
        public int Z { get; set; }
        ~MyClassWithFinalizer()
        {
            // an empty finalizer() can make GC a slower
            /* an instance of a class that contains a finalizer is
             * automatically promoted to a higher generation since
             * it cannot be collected in generation 0. */
        }
    }
    struct MyStruct
    {
        public int A { get; set; }
        public int B { get; set; }
        public int C { get; set; }
        public int X { get; set; }
        public int Y { get; set; }
        public int Z { get; set; }
    }
}