namespace RuntimeCompiler;

public class DriverClass
{
    public static void Main()
    {
        const string sourceCode = @"
namespace UtilityLibraries;

public static class NumberLibrary
{
    public static bool IsEven(int number)
    {
        return number % 2 == 0;
    }    
}";
        const string assemblyName = "mylib";
        var outputPath = Path.Combine(Path.GetTempPath(), $"{assemblyName}.dll");
        var rc = new RuntimeCompiler();
        rc.Compile(sourceCode, outputPath, assemblyName);
        Console.WriteLine($"DLL written to {outputPath}");
    }
}