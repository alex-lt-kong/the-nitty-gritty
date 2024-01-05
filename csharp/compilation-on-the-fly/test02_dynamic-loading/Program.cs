// Important note, it is mandatory to set:
// <PropertyGroup>
//     <PreserveCompilationContext>true</PreserveCompilationContext>
// </PropertyGroup>
// In .csproj file where RuntimeCompiler.Compile() will be called, otherwise defaultReferences will not work 
// Useful link: https://stackoverflow.com/questions/77730947/c-sharp-runtime-compilation-complains-the-type-object-is-defined-in-an-assembl/

using System.Reflection;



public class DriverClass
{
    public static void Main()
    {
        const string sourceCode = @"
namespace UtilityLibraries;

public static class NumberLibrary
{
    public static bool IsOdd(int number)
    {
        return number % 2 != 0;
    }
}
";
        const string assemblyName = "test02";
        var outputPath = Path.Combine(Path.GetTempPath(), $"{assemblyName}.dll");
        var rc = new RuntimeCompiler.RuntimeCompiler();
        rc.Compile(sourceCode, outputPath, assemblyName);
        Console.WriteLine($"DLL written to {outputPath}");
        var assembly = Assembly.LoadFrom(outputPath);
        
        
        const string typeName = "UtilityLibraries.NumberLibrary";
        const string methodName = "IsOdd";
        var type = assembly.GetType(typeName);
        if (type == null)
            throw new NotImplementedException($"Method {methodName} not implemented");

        var method = type.GetMethod(methodName, BindingFlags.Static | BindingFlags.Public);
        Console.WriteLine(method.Invoke(null, new object[] { 3 }));
        Console.WriteLine(method.Invoke(null, new object[] { 123 }));
        Console.WriteLine(method.Invoke(null, new object[] { 65536 }));
    }
}