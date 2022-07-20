using System.Diagnostics;

Random rnd = new Random();

void quicksort(int[] arr, int first, int last)
{

    int i, j, pivot, temp;

    if (first >= last) return;

    int idx = rnd.Next() % (last - first) + first;
    temp = arr[idx];
    arr[idx] = arr[first];
    arr[first] = temp;

    pivot = first;
    i = first;
    j = last;

    while (i < j)
    {
        while (arr[i] <= arr[pivot] && i < last) i++;
        while (arr[j] > arr[pivot]) j--;
        if (i < j)
        {
            temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    temp = arr[pivot];
    arr[pivot] = arr[j];
    arr[j] = temp;
    quicksort(arr, first, j - 1);
    quicksort(arr, j + 1, last);
}

int iterCount = 10;
long averageElapsedMs = 0;
int[] arr = new int[1_000_000];

for (int i = 0; i < iterCount; i++) {
    int count = 0;
    foreach (string line in System.IO.File.ReadLines($"..\\..\\..\\quicksort.in{i}"))
    {
        arr[count] = Int32.Parse(line);
        count++;
    }

    Stopwatch watch = new Stopwatch();
    watch.Start();
    quicksort(arr, 0, arr.Length-1);
    watch.Stop();
    averageElapsedMs += watch.ElapsedMilliseconds;
    Console.WriteLine($"{i}-th iteration: {watch.ElapsedMilliseconds:n0}ms");

    using (FileStream fs = File.OpenWrite($"..\\..\\..\\quicksort.out{i}"))
    {
        StreamWriter sw = new StreamWriter(fs);
        Array.ForEach(arr, sw.WriteLine);
    }
}
Console.WriteLine($"Average: {averageElapsedMs / iterCount:n0}ms");
