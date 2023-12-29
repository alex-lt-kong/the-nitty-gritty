using System.Collections.Immutable;
using Basic.Reference.Assemblies;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;

namespace RuntimeCompiler;

public static class RuntimeCompiler
{
    public static void Compile(string sourceCode, string outputPath, string assemblyName="MyAssembly")
    {
        var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
        var compilation = CSharpCompilation.Create(assemblyName)
            .WithOptions(new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary))
            .AddReferences(MetadataReference.CreateFromFile(typeof(object).Assembly.Location))
            .AddSyntaxTrees(syntaxTree);
    
        using var stream = new FileStream(outputPath, FileMode.Create);
        var result = compilation.Emit(stream);

        if (result.Success) return;
        var failures = result.Diagnostics
            .Where(diagnostic => diagnostic.IsWarningAsError || diagnostic.Severity == DiagnosticSeverity.Error);

        var errorMessages = string.Join(Environment.NewLine, failures.Select(diagnostic => diagnostic.GetMessage()));
        throw new Exception($"Compilation failed:{Environment.NewLine}{errorMessages}");
    }
}