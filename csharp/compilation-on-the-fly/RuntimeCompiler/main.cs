using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;

namespace RuntimeCompiler;

public class main
{
    public static void Main()
    {
        const string sourceCode = @"
namespace UtilityLibraries;

public static class StringLibrary
{
    public static bool StartsWithUpper(this string? str)
    {
        if (string.IsNullOrWhiteSpace(str))
            return false;

        char ch = str[0];
        return char.IsUpper(ch);
    }
}";
        const string outputPath = @"/tmp/mylib3.dll";
        RuntimeCompiler.Compile(sourceCode, outputPath, "mylib3");
    }
}