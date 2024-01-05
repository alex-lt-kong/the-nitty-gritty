// Important note, it is mandatory to set:
// <PropertyGroup>
//     <PreserveCompilationContext>true</PreserveCompilationContext>
// </PropertyGroup>
// In .csproj file where RuntimeCompiler.Compile() will be called, otherwise defaultReferences will not work 
// Useful link: https://stackoverflow.com/questions/77730947/c-sharp-runtime-compilation-complains-the-type-object-is-defined-in-an-assembl/


public static class DriverClass
{
    public static void Main()
    {
        const string sourceCode = @"
using System;

public class MyClass: IDemoInterface
{
    public int DemoMethod(int value)
    {
        return value / 2;
    }
}";
        const string assemblyName = "test03";
        var outputPath = Path.Combine(Path.GetTempPath(), $"{assemblyName}.dll");
        var rc = new RuntimeCompiler.RuntimeCompiler();
        var instance = rc.CompileWithInterface<IDemoInterface>(sourceCode, outputPath, assemblyName);
        Console.WriteLine($"DLL written to {outputPath}");
        Console.WriteLine(instance.DemoMethod(12345));
        Console.WriteLine(instance.DemoMethod(-2));
    }
}